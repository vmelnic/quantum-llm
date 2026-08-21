#include "expert/runtime/cuda/transformer_kernels.hpp"
#include "expert/runtime/expert_record.hpp"
#include "expert/runtime/model_artifact.hpp"
#include "expert/runtime/model_tensor_store.hpp"
#include "expert/runtime/sha256.hpp"

#include <cuda_runtime_api.h>
#include <immintrin.h>

#define NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <latch>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace er = expert::runtime;
namespace ec = expert::runtime::cuda;

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kAlignment = 4096U;
constexpr std::uint32_t kBlock = 32U;
constexpr std::uint32_t kPageCount = 3U;
constexpr std::string_view kCapability =
    "ffn.swiglu.dense.fp4-block32.v1";

#pragma pack(push, 1)
struct Section final {
  std::uint64_t offset{};
  std::uint64_t bytes{};
};

struct PageHeader final {
  char magic[8]{};
  std::uint32_t version{};
  std::uint32_t header_bytes{};
  std::uint32_t quant_abi{};
  std::uint32_t hidden{};
  std::uint32_t intermediate_begin{};
  std::uint32_t intermediate_count{};
  std::uint64_t stored_bytes{};
  std::uint64_t payload_bytes{};
  Section gate_weights;
  Section gate_scales;
  Section up_weights;
  Section up_scales;
  Section down_weights;
  Section down_scales;
  std::uint8_t payload_sha256[32]{};
  std::uint8_t reserved[80]{};
};

struct PageDirectoryEntry final {
  std::uint64_t offset{};
  std::uint64_t stored_bytes{};
  std::uint32_t intermediate_begin{};
  std::uint32_t intermediate_count{};
  std::uint32_t lane{};
  std::uint32_t reserved{};
};

struct ImageHeader final {
  char magic[8]{};
  std::uint32_t version{};
  std::uint32_t header_bytes{};
  std::uint32_t quant_abi{};
  std::uint32_t page_count{};
  std::uint32_t hidden{};
  std::uint32_t intermediate{};
  std::uint8_t model_content_sha256[32]{};
  std::uint32_t logical_operation{};
  std::uint32_t reserved0{};
  PageDirectoryEntry pages[8]{};
  std::uint8_t reserved[3768]{};
};
#pragma pack(pop)

static_assert(sizeof(PageHeader) == 256U);
static_assert(sizeof(PageDirectoryEntry) == 32U);
static_assert(sizeof(ImageHeader) == kAlignment);

struct Interval final {
  Clock::time_point begin;
  Clock::time_point end;
  [[nodiscard]] double milliseconds() const {
    return std::chrono::duration<double, std::milli>(end - begin).count();
  }
};

struct LaneResult final {
  Interval interval;
  double io_ms{};
  double h2d_ms{};
  std::uint64_t io_bytes{};
};

struct ErrorMetrics final {
  double maximum_absolute{};
  double relative_to_reference_maximum{};
  double rmse{};
  double cosine{};
};

[[noreturn]] void fail(std::string message) {
  throw std::runtime_error(std::move(message));
}

void require(bool condition, std::string_view message) {
  if (!condition) fail(std::string(message));
}

void status_check(const er::Status& status) {
  if (!status.ok()) fail(std::string(status.message()));
}

void cuda_check(cudaError_t error, std::string_view operation) {
  if (error != cudaSuccess)
    fail(std::string(operation) + ": " + cudaGetErrorString(error));
}

[[nodiscard]] std::uint64_t align_up(std::uint64_t value,
                                     std::uint64_t alignment) {
  require(alignment != 0U && value <=
              std::numeric_limits<std::uint64_t>::max() - (alignment - 1U),
          "invalid alignment request");
  return (value + alignment - 1U) / alignment * alignment;
}

[[nodiscard]] std::uint32_t padded32(std::uint32_t value) {
  return static_cast<std::uint32_t>(align_up(value, kBlock));
}

template <typename T>
class DeviceBuffer final {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    require(count != 0U, "zero CUDA buffer");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
               "allocate executable-page CUDA buffer");
  }
  ~DeviceBuffer() {
    if (data_ != nullptr) static_cast<void>(cudaFree(data_));
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  [[nodiscard]] T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }
  void upload(const void* source, std::size_t bytes,
              cudaStream_t stream = nullptr) {
    require(bytes <= this->bytes(), "CUDA upload exceeds allocation");
    cuda_check(cudaMemcpyAsync(data_, source, bytes, cudaMemcpyHostToDevice,
                               stream),
               "upload executable page");
  }
  void download(void* destination, std::size_t bytes,
                cudaStream_t stream = nullptr) const {
    require(bytes <= this->bytes(), "CUDA download exceeds allocation");
    cuda_check(cudaMemcpyAsync(destination, data_, bytes,
                               cudaMemcpyDeviceToHost, stream),
               "download executable page");
  }

 private:
  T* data_{};
  std::size_t count_{};
};

class Stream final {
 public:
  Stream() {
    cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "create executable-page CUDA stream");
  }
  ~Stream() {
    if (stream_ != nullptr) static_cast<void>(cudaStreamDestroy(stream_));
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
  void synchronize() const {
    cuda_check(cudaStreamSynchronize(stream_),
               "synchronize executable-page CUDA stream");
  }

 private:
  cudaStream_t stream_{};
};

struct SourceMatrix final {
  std::shared_ptr<const er::ImmutableModelTensor> tensor;
  const std::uint8_t* weights{};
  const std::uint8_t* scales{};
  std::uint32_t rows{};
  std::uint32_t columns{};
  std::uint32_t padded_columns{};
  std::size_t weight_bytes{};
  std::size_t scale_bytes{};
};

SourceMatrix source_matrix(
    std::shared_ptr<const er::ImmutableModelTensor> tensor) {
  require(tensor && tensor->value.valid() && tensor->encoding == "FP4_E2M1" &&
              tensor->quant_abi == er::kExpertQuantAbiFp4Block32 &&
              tensor->shape.size() == 2U,
          "SwiGLU tensor is not FP4 block-32");
  const auto rows = tensor->shape[0];
  const auto columns = tensor->shape[1];
  const auto padded = padded32(columns);
  const auto weights = static_cast<std::uint64_t>(rows) * padded / 2U;
  const auto scales = static_cast<std::uint64_t>(rows) * padded / kBlock;
  require(tensor->data_bytes == weights && tensor->scale_bytes == scales &&
              weights <= std::numeric_limits<std::size_t>::max() &&
              scales <= std::numeric_limits<std::size_t>::max(),
          "FP4 tensor has inconsistent storage geometry");
  return {tensor,
          reinterpret_cast<const std::uint8_t*>(tensor->value.data +
                                                tensor->data_offset),
          reinterpret_cast<const std::uint8_t*>(tensor->value.data +
                                                tensor->scale_offset),
          rows,
          columns,
          padded,
          static_cast<std::size_t>(weights),
          static_cast<std::size_t>(scales)};
}

const er::OperationProgramDescriptor& select_operation(
    const er::ModelDescriptor& model, std::optional<std::uint32_t> requested) {
  const auto found = std::find_if(
      model.operation_program.begin(), model.operation_program.end(),
      [&](const auto& operation) {
        return operation.capability == kCapability &&
               (!requested || operation.logical_operation == *requested);
      });
  require(found != model.operation_program.end(),
          "artifact has no selected FP4 dense SwiGLU operation");
  return *found;
}

SourceMatrix resolve_matrix(er::MappedModelTensorStore& store,
                            const er::OperationProgramDescriptor& operation,
                            std::string_view role) {
  const auto binding = operation.tensor_bindings.find(std::string(role));
  require(binding != operation.tensor_bindings.end(),
          "SwiGLU operation is missing a tensor role");
  auto result = store.resolve(binding->second);
  status_check(result.status);
  return source_matrix(std::move(result.tensor));
}

std::vector<std::byte> build_page(const SourceMatrix& gate,
                                  const SourceMatrix& up,
                                  const SourceMatrix& down,
                                  std::uint32_t begin,
                                  std::uint32_t count) {
  require(begin % kBlock == 0U && count % kBlock == 0U && count != 0U &&
              begin + count <= gate.rows,
          "invalid executable-page channel range");
  const auto gate_weight_bytes =
      static_cast<std::uint64_t>(count) * gate.padded_columns / 2U;
  const auto gate_scale_bytes =
      static_cast<std::uint64_t>(count) * gate.padded_columns / kBlock;
  const auto down_weight_bytes =
      static_cast<std::uint64_t>(down.rows) * count / 2U;
  const auto down_scale_bytes =
      static_cast<std::uint64_t>(down.rows) * count / kBlock;
  std::uint64_t cursor = sizeof(PageHeader);
  const auto assign = [&](Section& section, std::uint64_t bytes) {
    section = {cursor, bytes};
    cursor += bytes;
  };
  PageHeader header{};
  std::memcpy(header.magic, "SWPG0001", 8U);
  header.version = 1U;
  header.header_bytes = sizeof(PageHeader);
  header.quant_abi = er::kExpertQuantAbiFp4Block32;
  header.hidden = gate.columns;
  header.intermediate_begin = begin;
  header.intermediate_count = count;
  assign(header.gate_weights, gate_weight_bytes);
  assign(header.gate_scales, gate_scale_bytes);
  assign(header.up_weights, gate_weight_bytes);
  assign(header.up_scales, gate_scale_bytes);
  assign(header.down_weights, down_weight_bytes);
  assign(header.down_scales, down_scale_bytes);
  header.payload_bytes = cursor;
  header.stored_bytes = align_up(cursor, kAlignment);
  require(header.stored_bytes <= std::numeric_limits<std::size_t>::max(),
          "executable page exceeds address space");
  std::vector<std::byte> page(static_cast<std::size_t>(header.stored_bytes));
  std::memcpy(page.data(), &header, sizeof(header));
  auto* destination = reinterpret_cast<std::uint8_t*>(page.data());
  const auto gate_row_bytes = gate.padded_columns / 2U;
  const auto gate_row_scales = gate.padded_columns / kBlock;
  std::memcpy(destination + header.gate_weights.offset,
              gate.weights + static_cast<std::size_t>(begin) * gate_row_bytes,
              static_cast<std::size_t>(header.gate_weights.bytes));
  std::memcpy(destination + header.gate_scales.offset,
              gate.scales + static_cast<std::size_t>(begin) * gate_row_scales,
              static_cast<std::size_t>(header.gate_scales.bytes));
  std::memcpy(destination + header.up_weights.offset,
              up.weights + static_cast<std::size_t>(begin) * gate_row_bytes,
              static_cast<std::size_t>(header.up_weights.bytes));
  std::memcpy(destination + header.up_scales.offset,
              up.scales + static_cast<std::size_t>(begin) * gate_row_scales,
              static_cast<std::size_t>(header.up_scales.bytes));
  const auto source_down_row_bytes = down.padded_columns / 2U;
  const auto source_down_row_scales = down.padded_columns / kBlock;
  const auto target_down_row_bytes = count / 2U;
  const auto target_down_row_scales = count / kBlock;
  for (std::uint32_t row = 0U; row < down.rows; ++row) {
    std::memcpy(destination + header.down_weights.offset +
                    static_cast<std::size_t>(row) * target_down_row_bytes,
                down.weights +
                    static_cast<std::size_t>(row) * source_down_row_bytes +
                    begin / 2U,
                target_down_row_bytes);
    std::memcpy(destination + header.down_scales.offset +
                    static_cast<std::size_t>(row) * target_down_row_scales,
                down.scales +
                    static_cast<std::size_t>(row) * source_down_row_scales +
                    begin / kBlock,
                target_down_row_scales);
  }
  require(std::memcmp(destination + header.gate_weights.offset,
                      gate.weights +
                          static_cast<std::size_t>(begin) * gate_row_bytes,
                      static_cast<std::size_t>(header.gate_weights.bytes)) ==
              0 &&
              std::memcmp(destination + header.up_weights.offset,
                          up.weights +
                              static_cast<std::size_t>(begin) * gate_row_bytes,
                          static_cast<std::size_t>(header.up_weights.bytes)) ==
                  0,
          "repacked gate/up FP4 bytes differ from the source tensor");
  const auto payload = std::span(page).subspan(
      sizeof(PageHeader), static_cast<std::size_t>(header.payload_bytes) -
                              sizeof(PageHeader));
  const auto digest = er::sha256(payload);
  std::memcpy(reinterpret_cast<PageHeader*>(page.data())->payload_sha256,
              digest.data(), digest.size());
  return page;
}

const PageHeader& validate_page(std::span<const std::byte> page,
                                const PageDirectoryEntry& directory) {
  require(page.size() == directory.stored_bytes &&
              page.size() >= sizeof(PageHeader),
          "executable page size mismatch");
  const auto& header = *reinterpret_cast<const PageHeader*>(page.data());
  require(std::memcmp(header.magic, "SWPG0001", 8U) == 0 &&
              header.version == 1U &&
              header.header_bytes == sizeof(PageHeader) &&
              header.quant_abi == er::kExpertQuantAbiFp4Block32 &&
              header.stored_bytes == page.size() &&
              header.payload_bytes >= sizeof(PageHeader) &&
              header.payload_bytes <= header.stored_bytes &&
              header.intermediate_begin == directory.intermediate_begin &&
              header.intermediate_count == directory.intermediate_count,
          "invalid executable page header");
  const auto within = [&](const Section& section) {
    return section.offset >= sizeof(PageHeader) &&
           section.offset <= header.payload_bytes &&
           section.bytes <= header.payload_bytes - section.offset;
  };
  require(within(header.gate_weights) && within(header.gate_scales) &&
              within(header.up_weights) && within(header.up_scales) &&
              within(header.down_weights) && within(header.down_scales),
          "executable page section exceeds payload");
  const auto digest = er::sha256(page.subspan(
      sizeof(PageHeader), static_cast<std::size_t>(header.payload_bytes) -
                              sizeof(PageHeader)));
  er::Sha256Digest expected{};
  std::memcpy(expected.data(), header.payload_sha256, expected.size());
  require(er::constant_time_equal(digest, expected),
          "executable page checksum mismatch");
  return header;
}

ImageHeader write_image(const std::filesystem::path& path,
                        const er::ModelDescriptor& model,
                        const er::OperationProgramDescriptor& operation,
                        const SourceMatrix& gate, const SourceMatrix& up,
                        const SourceMatrix& down) {
  require(gate.rows == up.rows && gate.columns == up.columns &&
              down.rows == gate.columns && down.columns == gate.rows &&
              gate.rows % kBlock == 0U && gate.columns % kBlock == 0U,
          "SwiGLU tensor geometry cannot be partitioned exactly");
  ImageHeader image{};
  std::memcpy(image.magic, "SWIMG001", 8U);
  image.version = 1U;
  image.header_bytes = sizeof(ImageHeader);
  image.quant_abi = er::kExpertQuantAbiFp4Block32;
  image.page_count = kPageCount;
  image.hidden = gate.columns;
  image.intermediate = gate.rows;
  std::memcpy(image.model_content_sha256, model.content_hash.data(),
              model.content_hash.size());
  image.logical_operation = operation.logical_operation;

  const auto blocks = gate.rows / kBlock;
  const std::array<std::uint32_t, kPageCount> block_counts{
      blocks / 3U, blocks / 3U, blocks - 2U * (blocks / 3U)};
  require(block_counts[0] != 0U,
          "SwiGLU intermediate is too small for three pages");
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  require(static_cast<bool>(output), "cannot create executable model image");
  output.write(reinterpret_cast<const char*>(&image), sizeof(image));
  std::uint64_t offset = sizeof(ImageHeader);
  std::uint32_t begin = 0U;
  for (std::uint32_t index = 0U; index < kPageCount; ++index) {
    const auto count = block_counts[index] * kBlock;
    auto page = build_page(gate, up, down, begin, count);
    image.pages[index] = {offset, page.size(), begin, count, index, 0U};
    output.write(reinterpret_cast<const char*>(page.data()),
                 static_cast<std::streamsize>(page.size()));
    require(static_cast<bool>(output), "cannot write executable model page");
    offset += page.size();
    begin += count;
  }
  require(begin == gate.rows, "executable pages do not cover SwiGLU width");
  output.seekp(0U);
  output.write(reinterpret_cast<const char*>(&image), sizeof(image));
  output.close();
  require(static_cast<bool>(output), "cannot finalize executable model image");
  const auto handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                  FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  require(handle != INVALID_HANDLE_VALUE,
          "cannot reopen executable image for durable flush");
  const auto flushed = FlushFileBuffers(handle);
  CloseHandle(handle);
  require(flushed != 0, "cannot flush executable image to storage");
  return image;
}

std::vector<std::byte> read_page_buffered(
    const std::filesystem::path& path, const PageDirectoryEntry& entry) {
  require(entry.stored_bytes <= std::numeric_limits<std::size_t>::max(),
          "page exceeds address space");
  std::ifstream input(path, std::ios::binary);
  require(static_cast<bool>(input), "cannot open executable image");
  input.seekg(static_cast<std::streamoff>(entry.offset));
  std::vector<std::byte> page(static_cast<std::size_t>(entry.stored_bytes));
  input.read(reinterpret_cast<char*>(page.data()),
             static_cast<std::streamsize>(page.size()));
  require(input.gcount() == static_cast<std::streamsize>(page.size()),
          "truncated executable image page");
  static_cast<void>(validate_page(page, entry));
  return page;
}

class DirectPage final {
 public:
  DirectPage(const std::filesystem::path& path,
             const PageDirectoryEntry& entry)
      : entry_(entry) {
    require(entry.offset % kAlignment == 0U &&
                entry.stored_bytes % kAlignment == 0U &&
                entry.stored_bytes <= std::numeric_limits<DWORD>::max(),
            "direct-I/O page is not sector aligned");
    file_ = CreateFileW(path.c_str(), GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        nullptr, OPEN_EXISTING,
                        FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED |
                            FILE_FLAG_SEQUENTIAL_SCAN,
                        nullptr);
    require(file_ != INVALID_HANDLE_VALUE,
            "cannot open executable image with unbuffered I/O");
    buffer_ = VirtualAlloc(nullptr, static_cast<SIZE_T>(entry.stored_bytes),
                           MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    require(buffer_ != nullptr, "cannot allocate aligned direct-I/O buffer");
    event_ = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    require(event_ != nullptr, "cannot create direct-I/O completion event");
  }
  ~DirectPage() {
    if (event_ != nullptr) CloseHandle(event_);
    if (buffer_ != nullptr) VirtualFree(buffer_, 0U, MEM_RELEASE);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
  }
  DirectPage(const DirectPage&) = delete;
  DirectPage& operator=(const DirectPage&) = delete;
  std::span<const std::byte> read(double& milliseconds) {
    OVERLAPPED overlapped{};
    overlapped.Offset = static_cast<DWORD>(entry_.offset & 0xffffffffULL);
    overlapped.OffsetHigh = static_cast<DWORD>(entry_.offset >> 32U);
    overlapped.hEvent = event_;
    ResetEvent(event_);
    const auto started = Clock::now();
    const auto immediate = ReadFile(file_, buffer_,
                                    static_cast<DWORD>(entry_.stored_bytes),
                                    nullptr, &overlapped);
    if (immediate == 0 && GetLastError() != ERROR_IO_PENDING)
      fail("unbuffered executable-page read could not be submitted");
    const auto wait = WaitForSingleObject(event_, INFINITE);
    require(wait == WAIT_OBJECT_0,
            "unbuffered executable-page read did not complete");
    DWORD transferred{};
    require(GetOverlappedResult(file_, &overlapped, &transferred, FALSE) != 0 &&
                transferred == static_cast<DWORD>(entry_.stored_bytes),
            "unbuffered executable-page read was truncated");
    milliseconds = std::chrono::duration<double, std::milli>(Clock::now() -
                                                              started)
                       .count();
    const auto page = std::span(
        reinterpret_cast<const std::byte*>(buffer_),
        static_cast<std::size_t>(entry_.stored_bytes));
    static_cast<void>(validate_page(page, entry_));
    return page;
  }

 private:
  PageDirectoryEntry entry_;
  HANDLE file_{INVALID_HANDLE_VALUE};
  HANDLE event_{};
  void* buffer_{};
};

float decode_scale(std::uint8_t code) noexcept {
  const std::uint32_t bits =
      code == 0U ? 0x00400000U : std::uint32_t{code} << 23U;
  return std::bit_cast<float>(bits);
}

__m256i dot_fp4_q8_lanes(const std::uint8_t* packed,
                         const std::int8_t* activation) noexcept {
  const auto source =
      _mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
  const auto mask = _mm_set1_epi8(0x0f);
  const auto low = _mm_and_si128(source, mask);
  const auto high = _mm_and_si128(_mm_srli_epi16(source, 4), mask);
  const auto indices = _mm256_set_m128i(_mm_unpackhi_epi8(low, high),
                                        _mm_unpacklo_epi8(low, high));
  const auto table = _mm256_broadcastsi128_si256(_mm_setr_epi8(
      0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12));
  const auto weights = _mm256_shuffle_epi8(table, indices);
  const auto values = _mm256_loadu_si256(
      reinterpret_cast<const __m256i*>(activation));
  const auto signed_weights = _mm256_sign_epi8(weights, values);
  const auto magnitudes = _mm256_abs_epi8(values);
  return _mm256_madd_epi16(
      _mm256_maddubs_epi16(magnitudes, signed_weights),
      _mm256_set1_epi16(1));
}

float horizontal_sum(__m256 value) noexcept {
  auto lanes = _mm_add_ps(_mm256_castps256_ps128(value),
                          _mm256_extractf128_ps(value, 1));
  lanes = _mm_hadd_ps(lanes, lanes);
  lanes = _mm_hadd_ps(lanes, lanes);
  return _mm_cvtss_f32(lanes);
}

float packed_dot(const std::uint8_t* weights, const std::uint8_t* scales,
                 const std::int8_t* activation, float activation_scale,
                 std::uint32_t row, std::uint32_t columns) noexcept {
  const auto blocks = columns / kBlock;
  const auto* row_weights =
      weights + static_cast<std::size_t>(row) * columns / 2U;
  const auto* row_scales =
      scales + static_cast<std::size_t>(row) * blocks;
  auto lanes = _mm256_setzero_ps();
  for (std::uint32_t block = 0U; block < blocks; ++block) {
    const auto products = dot_fp4_q8_lanes(
        row_weights + static_cast<std::size_t>(block) * 16U,
        activation + static_cast<std::size_t>(block) * kBlock);
    lanes = _mm256_add_ps(
        lanes, _mm256_mul_ps(_mm256_cvtepi32_ps(products),
                             _mm256_set1_ps(decode_scale(row_scales[block]))));
  }
  return horizontal_sum(lanes) * activation_scale * 0.5F;
}

template <typename Function>
void parallel_for(std::uint32_t count, Function function) {
  const auto worker_count = std::min(
      count, std::max(1U, std::thread::hardware_concurrency()));
  std::atomic<std::uint32_t> next{};
  std::vector<std::thread> workers;
  workers.reserve(worker_count);
  for (std::uint32_t worker = 0U; worker < worker_count; ++worker) {
    workers.emplace_back([&] {
      for (;;) {
        const auto index = next.fetch_add(1U, std::memory_order_relaxed);
        if (index >= count) return;
        function(index);
      }
    });
  }
  for (auto& worker : workers) worker.join();
}

void cpu_gate_up(const PageHeader& page, const std::byte* bytes,
                 std::span<const std::int8_t> input, float input_scale,
                 std::span<float> intermediate) {
  require(input.size() == page.hidden &&
              intermediate.size() == page.intermediate_count,
          "CPU gate/up span mismatch");
  const auto* gate = reinterpret_cast<const std::uint8_t*>(
      bytes + page.gate_weights.offset);
  const auto* gate_scales = reinterpret_cast<const std::uint8_t*>(
      bytes + page.gate_scales.offset);
  const auto* up = reinterpret_cast<const std::uint8_t*>(
      bytes + page.up_weights.offset);
  const auto* up_scales = reinterpret_cast<const std::uint8_t*>(
      bytes + page.up_scales.offset);
  parallel_for(page.intermediate_count, [&](std::uint32_t row) {
    const auto gate_value = packed_dot(gate, gate_scales, input.data(),
                                       input_scale, row, page.hidden);
    const auto up_value = packed_dot(up, up_scales, input.data(), input_scale,
                                     row, page.hidden);
    intermediate[row] =
        (gate_value / (1.0F + std::exp(-gate_value))) * up_value;
  });
}

void cpu_down(const PageHeader& page, const std::byte* bytes,
              std::span<const std::int8_t> intermediate,
              float intermediate_scale, std::span<float> output) {
  require(intermediate.size() == page.intermediate_count &&
              output.size() == page.hidden,
          "CPU down span mismatch");
  const auto* weights = reinterpret_cast<const std::uint8_t*>(
      bytes + page.down_weights.offset);
  const auto* scales = reinterpret_cast<const std::uint8_t*>(
      bytes + page.down_scales.offset);
  parallel_for(page.hidden, [&](std::uint32_t row) {
    output[row] = packed_dot(weights, scales, intermediate.data(),
                             intermediate_scale, row,
                             page.intermediate_count);
  });
}

float quantize_q8(std::span<const float> input,
                  std::span<std::int8_t> output) {
  require(input.size() == output.size() && !input.empty(),
          "Q8 quantization span mismatch");
  float maximum = 0.0F;
  for (const auto value : input) maximum = std::max(maximum, std::abs(value));
  const auto scale = maximum > 0.0F ? maximum / 127.0F : 1.0F;
  for (std::size_t index = 0U; index < input.size(); ++index) {
    const auto rounded =
        static_cast<int>(std::nearbyint(input[index] / scale));
    output[index] = static_cast<std::int8_t>(
        std::clamp(rounded, -127, 127));
  }
  return scale;
}

ec::Fp4Block32Matrix device_matrix(const PageHeader& page,
                                   const std::byte* base,
                                   const Section& weights,
                                   const Section& scales,
                                   std::uint32_t rows,
                                   std::uint32_t columns) {
  return {reinterpret_cast<const std::uint8_t*>(base + weights.offset),
          reinterpret_cast<const std::uint8_t*>(base + scales.offset), rows,
          columns, padded32(columns)};
}

struct DevicePage final {
  explicit DevicePage(std::size_t stored_bytes, std::uint32_t channels)
      : record(stored_bytes), gate(channels), up(channels),
        intermediate(channels), q_intermediate(channels),
        q_intermediate_scale(1U) {}

  DeviceBuffer<std::byte> record;
  DeviceBuffer<float> gate;
  DeviceBuffer<float> up;
  DeviceBuffer<float> intermediate;
  DeviceBuffer<std::int8_t> q_intermediate;
  DeviceBuffer<float> q_intermediate_scale;
  std::unique_ptr<DeviceBuffer<float>> output;
};

void allocate_page_output(DevicePage& page, std::uint32_t hidden) {
  page.output = std::make_unique<DeviceBuffer<float>>(hidden);
}

void gpu_gate_up(const PageHeader& page, DevicePage& device,
                 const DeviceBuffer<std::int8_t>& q_input,
                 const DeviceBuffer<float>& q_input_scale,
                 cudaStream_t stream) {
  const auto* base = device.record.data();
  status_check(ec::fp4_gemv_q8_batch(
      device_matrix(page, base, page.gate_weights, page.gate_scales,
                    page.intermediate_count, page.hidden),
      q_input.data(), q_input_scale.data(), device.gate.data(), 1U, stream));
  status_check(ec::fp4_gemv_q8_batch(
      device_matrix(page, base, page.up_weights, page.up_scales,
                    page.intermediate_count, page.hidden),
      q_input.data(), q_input_scale.data(), device.up.data(), 1U, stream));
  status_check(ec::silu_product(device.gate.data(), device.up.data(),
                                device.intermediate.data(),
                                page.intermediate_count, stream));
}

void gpu_down(const PageHeader& page, DevicePage& device,
              std::span<const std::int8_t> q_intermediate,
              float intermediate_scale, cudaStream_t stream) {
  require(device.output != nullptr, "GPU page has no output allocation");
  device.q_intermediate.upload(q_intermediate.data(), q_intermediate.size(),
                               stream);
  device.q_intermediate_scale.upload(&intermediate_scale,
                                     sizeof(intermediate_scale), stream);
  const auto* base = device.record.data();
  status_check(ec::fp4_gemv_q8_batch(
      device_matrix(page, base, page.down_weights, page.down_scales,
                    page.hidden, page.intermediate_count),
      device.q_intermediate.data(), device.q_intermediate_scale.data(),
      device.output->data(), 1U, stream));
}

struct Baseline final {
  Baseline(std::uint32_t hidden_value, std::uint32_t intermediate_value,
           const SourceMatrix& gate_source, const SourceMatrix& up_source,
           const SourceMatrix& down_source)
      : hidden(hidden_value), intermediate_size(intermediate_value),
        gate_weights(gate_source.weight_bytes),
        gate_scales(gate_source.scale_bytes), up_weights(up_source.weight_bytes),
        up_scales(up_source.scale_bytes), down_weights(down_source.weight_bytes),
        down_scales(down_source.scale_bytes), input(hidden), q_input(hidden),
        q_input_scale(1U), gate(intermediate_size), up(intermediate_size),
        intermediate(intermediate_size), q_intermediate(intermediate_size),
        q_intermediate_scale(1U), output(hidden) {
    gate_weights.upload(gate_source.weights, gate_source.weight_bytes);
    gate_scales.upload(gate_source.scales, gate_source.scale_bytes);
    up_weights.upload(up_source.weights, up_source.weight_bytes);
    up_scales.upload(up_source.scales, up_source.scale_bytes);
    down_weights.upload(down_source.weights, down_source.weight_bytes);
    down_scales.upload(down_source.scales, down_source.scale_bytes);
    cuda_check(cudaDeviceSynchronize(), "upload resident SwiGLU reference");
  }

  void execute(cudaStream_t stream) {
    status_check(ec::quantize_q8_batch(input.data(), q_input.data(),
                                       q_input_scale.data(), 1U, hidden,
                                       hidden, stream));
    status_check(ec::fp4_gemv_q8_batch(
        {gate_weights.data(), gate_scales.data(), intermediate_size, hidden,
         hidden},
        q_input.data(), q_input_scale.data(), gate.data(), 1U, stream));
    status_check(ec::fp4_gemv_q8_batch(
        {up_weights.data(), up_scales.data(), intermediate_size, hidden,
         hidden},
        q_input.data(), q_input_scale.data(), up.data(), 1U, stream));
    status_check(ec::silu_product(gate.data(), up.data(), intermediate.data(),
                                  intermediate_size, stream));
    status_check(ec::quantize_q8_batch(
        intermediate.data(), q_intermediate.data(),
        q_intermediate_scale.data(), 1U, intermediate_size,
        intermediate_size, stream));
    status_check(ec::fp4_gemv_q8_batch(
        {down_weights.data(), down_scales.data(), hidden, intermediate_size,
         intermediate_size},
        q_intermediate.data(), q_intermediate_scale.data(), output.data(), 1U,
        stream));
  }

  std::uint32_t hidden{};
  std::uint32_t intermediate_size{};
  DeviceBuffer<std::uint8_t> gate_weights;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_weights;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<std::uint8_t> down_weights;
  DeviceBuffer<std::uint8_t> down_scales;
  DeviceBuffer<float> input;
  DeviceBuffer<std::int8_t> q_input;
  DeviceBuffer<float> q_input_scale;
  DeviceBuffer<float> gate;
  DeviceBuffer<float> up;
  DeviceBuffer<float> intermediate;
  DeviceBuffer<std::int8_t> q_intermediate;
  DeviceBuffer<float> q_intermediate_scale;
  DeviceBuffer<float> output;
};

ErrorMetrics compare(std::span<const float> actual,
                     std::span<const float> reference) {
  require(actual.size() == reference.size() && !actual.empty(),
          "comparison span mismatch");
  double squared = 0.0;
  double dot = 0.0;
  double actual_norm = 0.0;
  double reference_norm = 0.0;
  double reference_maximum = 0.0;
  ErrorMetrics result{};
  for (std::size_t index = 0U; index < actual.size(); ++index) {
    require(std::isfinite(actual[index]) && std::isfinite(reference[index]),
            "non-finite numerical comparison");
    const auto delta = std::abs(static_cast<double>(actual[index]) -
                                reference[index]);
    result.maximum_absolute = std::max(result.maximum_absolute, delta);
    reference_maximum =
        std::max(reference_maximum, std::abs(static_cast<double>(reference[index])));
    squared += delta * delta;
    dot += static_cast<double>(actual[index]) * reference[index];
    actual_norm += static_cast<double>(actual[index]) * actual[index];
    reference_norm += static_cast<double>(reference[index]) * reference[index];
  }
  result.relative_to_reference_maximum =
      reference_maximum > 0.0 ? result.maximum_absolute / reference_maximum
                              : result.maximum_absolute;
  result.rmse = std::sqrt(squared / actual.size());
  result.cosine = dot / std::sqrt(actual_norm * reference_norm);
  return result;
}

double overlap_ms(const std::array<Interval, kPageCount>& intervals) {
  auto begin = intervals.front().begin;
  auto end = intervals.front().end;
  for (const auto& interval : intervals) {
    begin = std::max(begin, interval.begin);
    end = std::min(end, interval.end);
  }
  if (end <= begin) return 0.0;
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

void json_string(std::ostream& output, std::string_view value) {
  output << '"';
  for (const auto character : value) {
    if (character == '"' || character == '\\') output << '\\';
    output << character;
  }
  output << '"';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3 || argc > 4) {
      std::cerr << "usage: expert-executable-swiglu-page <artifact-root> "
                   "<image-path> [logical-operation]\n";
      return 64;
    }
    const std::filesystem::path artifact_root(argv[1]);
    const std::filesystem::path image_path(argv[2]);
    const auto requested =
        argc == 4 ? std::optional<std::uint32_t>(
                        static_cast<std::uint32_t>(std::stoul(argv[3])))
                  : std::nullopt;

    ImageHeader image{};
    std::string gate_name;
    std::string up_name;
    std::string down_name;
    std::unique_ptr<Baseline> baseline;
    {
      er::ModelArtifact artifact;
      status_check(er::ModelArtifact::load(artifact_root, artifact));
      const auto& operation = select_operation(artifact.model(), requested);
      er::MappedModelTensorStore store;
      status_check(er::MappedModelTensorStore::create(artifact, store));
      auto gate = resolve_matrix(store, operation, "gate_projection");
      auto up = resolve_matrix(store, operation, "up_projection");
      auto down = resolve_matrix(store, operation, "down_projection");
      gate_name = gate.tensor->name;
      up_name = up.tensor->name;
      down_name = down.tensor->name;
      image = write_image(image_path, artifact.model(), operation, gate, up,
                          down);
      baseline = std::make_unique<Baseline>(image.hidden, image.intermediate,
                                            gate, up, down);
    }

    auto gpu_page_bytes = read_page_buffered(image_path, image.pages[0]);
    auto cpu_page_bytes = read_page_buffered(image_path, image.pages[1]);
    const auto& gpu_header = validate_page(gpu_page_bytes, image.pages[0]);
    const auto& cpu_header = validate_page(cpu_page_bytes, image.pages[1]);
    PageHeader nvme_header{};

    DevicePage gpu_page(image.pages[0].stored_bytes,
                        image.pages[0].intermediate_count);
    DevicePage nvme_page(image.pages[2].stored_bytes,
                         image.pages[2].intermediate_count);
    allocate_page_output(gpu_page, image.hidden);
    allocate_page_output(nvme_page, image.hidden);
    Stream baseline_stream;
    Stream resident_stream;
    Stream streamed_stream;
    gpu_page.record.upload(gpu_page_bytes.data(), gpu_page_bytes.size(),
                           resident_stream.get());
    resident_stream.synchronize();

    std::vector<float> input(image.hidden);
    for (std::size_t index = 0U; index < input.size(); ++index) {
      const auto phase = static_cast<float>((index * 17U) % 4093U) * 0.0031F;
      input[index] = std::sin(phase) * 0.72F +
                     std::cos(phase * 0.37F) * 0.19F;
    }
    baseline->input.upload(input.data(), input.size() * sizeof(float),
                           baseline_stream.get());
    baseline->execute(baseline_stream.get());
    baseline_stream.synchronize();
    cudaEvent_t baseline_start{}, baseline_stop{};
    cuda_check(cudaEventCreate(&baseline_start), "create baseline start event");
    cuda_check(cudaEventCreate(&baseline_stop), "create baseline stop event");
    cuda_check(cudaEventRecord(baseline_start, baseline_stream.get()),
               "record baseline start");
    baseline->execute(baseline_stream.get());
    cuda_check(cudaEventRecord(baseline_stop, baseline_stream.get()),
               "record baseline stop");
    cuda_check(cudaEventSynchronize(baseline_stop), "wait for baseline");
    float baseline_ms{};
    cuda_check(cudaEventElapsedTime(&baseline_ms, baseline_start, baseline_stop),
               "measure resident SwiGLU baseline");
    cudaEventDestroy(baseline_start);
    cudaEventDestroy(baseline_stop);

    std::vector<float> reference_output(image.hidden);
    std::vector<float> reference_intermediate(image.intermediate);
    std::vector<std::int8_t> q_input(image.hidden);
    float input_scale{};
    baseline->output.download(reference_output.data(),
                              reference_output.size() * sizeof(float),
                              baseline_stream.get());
    baseline->intermediate.download(
        reference_intermediate.data(),
        reference_intermediate.size() * sizeof(float), baseline_stream.get());
    baseline->q_input.download(q_input.data(), q_input.size(),
                               baseline_stream.get());
    baseline->q_input_scale.download(&input_scale, sizeof(input_scale),
                                     baseline_stream.get());
    baseline_stream.synchronize();

    std::vector<float> hybrid_intermediate(image.intermediate);
    DirectPage direct_page(image_path, image.pages[2]);
    std::latch gate_start(1U);
    auto resident_future = std::async(std::launch::async, [&] {
      LaneResult result;
      gate_start.wait();
      result.interval.begin = Clock::now();
      gpu_gate_up(gpu_header, gpu_page, baseline->q_input,
                  baseline->q_input_scale, resident_stream.get());
      resident_stream.synchronize();
      result.interval.end = Clock::now();
      return result;
    });
    auto cpu_future = std::async(std::launch::async, [&] {
      LaneResult result;
      gate_start.wait();
      result.interval.begin = Clock::now();
      cpu_gate_up(
          cpu_header, cpu_page_bytes.data(), q_input, input_scale,
          std::span(hybrid_intermediate)
              .subspan(cpu_header.intermediate_begin,
                       cpu_header.intermediate_count));
      result.interval.end = Clock::now();
      return result;
    });
    auto streamed_future = std::async(std::launch::async, [&] {
      LaneResult result;
      gate_start.wait();
      result.interval.begin = Clock::now();
      auto page = direct_page.read(result.io_ms);
      result.io_bytes = page.size();
      const auto& checked = validate_page(page, image.pages[2]);
      nvme_header = checked;
      const auto h2d_started = Clock::now();
      nvme_page.record.upload(page.data(), page.size(), streamed_stream.get());
      gpu_gate_up(nvme_header, nvme_page, baseline->q_input,
                  baseline->q_input_scale, streamed_stream.get());
      streamed_stream.synchronize();
      result.h2d_ms =
          std::chrono::duration<double, std::milli>(Clock::now() - h2d_started)
              .count();
      result.interval.end = Clock::now();
      return result;
    });
    const auto gate_wall_started = Clock::now();
    gate_start.count_down();
    std::array<LaneResult, kPageCount> gate_lanes{
        resident_future.get(), cpu_future.get(), streamed_future.get()};
    const auto gate_wall_ms =
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  gate_wall_started)
            .count();
    gpu_page.intermediate.download(
        hybrid_intermediate.data() + gpu_header.intermediate_begin,
        static_cast<std::size_t>(gpu_header.intermediate_count) * sizeof(float),
        resident_stream.get());
    nvme_page.intermediate.download(
        hybrid_intermediate.data() + nvme_header.intermediate_begin,
        static_cast<std::size_t>(nvme_header.intermediate_count) * sizeof(float),
        streamed_stream.get());
    resident_stream.synchronize();
    streamed_stream.synchronize();

    std::vector<std::int8_t> q_intermediate(image.intermediate);
    const auto intermediate_scale =
        quantize_q8(hybrid_intermediate, q_intermediate);
    std::vector<float> cpu_partial(image.hidden);
    std::vector<float> gpu_partial(image.hidden);
    std::vector<float> nvme_partial(image.hidden);
    const auto down_started = Clock::now();
    auto gpu_down_future = std::async(std::launch::async, [&] {
      gpu_down(gpu_header, gpu_page,
               std::span(q_intermediate)
                   .subspan(gpu_header.intermediate_begin,
                            gpu_header.intermediate_count),
               intermediate_scale, resident_stream.get());
      resident_stream.synchronize();
    });
    auto cpu_down_future = std::async(std::launch::async, [&] {
      cpu_down(cpu_header, cpu_page_bytes.data(),
               std::span(q_intermediate)
                   .subspan(cpu_header.intermediate_begin,
                            cpu_header.intermediate_count),
               intermediate_scale, cpu_partial);
    });
    auto nvme_down_future = std::async(std::launch::async, [&] {
      gpu_down(nvme_header, nvme_page,
               std::span(q_intermediate)
                   .subspan(nvme_header.intermediate_begin,
                            nvme_header.intermediate_count),
               intermediate_scale, streamed_stream.get());
      streamed_stream.synchronize();
    });
    gpu_down_future.get();
    cpu_down_future.get();
    nvme_down_future.get();
    const auto down_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - down_started)
            .count();
    gpu_page.output->download(gpu_partial.data(),
                              gpu_partial.size() * sizeof(float),
                              resident_stream.get());
    nvme_page.output->download(nvme_partial.data(),
                               nvme_partial.size() * sizeof(float),
                               streamed_stream.get());
    resident_stream.synchronize();
    streamed_stream.synchronize();
    std::vector<float> hybrid_output(image.hidden);
    for (std::size_t index = 0U; index < hybrid_output.size(); ++index)
      hybrid_output[index] =
          (gpu_partial[index] + cpu_partial[index]) + nvme_partial[index];

    const auto output_error = compare(hybrid_output, reference_output);
    const auto intermediate_error =
        compare(hybrid_intermediate, reference_intermediate);
    const std::array<Interval, kPageCount> intervals{
        gate_lanes[0].interval, gate_lanes[1].interval,
        gate_lanes[2].interval};
    const auto simultaneous_ms = overlap_ms(intervals);
    const auto resident_weight_bytes =
        baseline->gate_weights.bytes() + baseline->gate_scales.bytes() +
        baseline->up_weights.bytes() + baseline->up_scales.bytes() +
        baseline->down_weights.bytes() + baseline->down_scales.bytes();
    const auto paged_vram_weight_bytes = image.pages[0].stored_bytes +
                                         image.pages[2].stored_bytes;
    const auto valid = output_error.relative_to_reference_maximum < 1.0e-3 &&
                       output_error.cosine > 0.999999 &&
                       gate_lanes[2].io_bytes == image.pages[2].stored_bytes &&
                       simultaneous_ms > 0.0;

    std::cout << std::setprecision(12)
              << "{\"schema_version\":1,\"valid\":"
              << (valid ? "true" : "false")
              << ",\"artifact\":";
    json_string(std::cout, artifact_root.generic_string());
    std::cout << ",\"image\":";
    json_string(std::cout, image_path.generic_string());
    std::cout << ",\"capability\":";
    json_string(std::cout, kCapability);
    std::cout << ",\"logical_operation\":" << image.logical_operation
              << ",\"gate_tensor\":";
    json_string(std::cout, gate_name);
    std::cout << ",\"up_tensor\":";
    json_string(std::cout, up_name);
    std::cout << ",\"down_tensor\":";
    json_string(std::cout, down_name);
    std::cout << ",\"encoding\":\"FP4_E2M1\""
              << ",\"scale_encoding\":\"UE8M0\""
              << ",\"quant_abi\":" << image.quant_abi
              << ",\"hidden\":" << image.hidden
              << ",\"intermediate\":" << image.intermediate
              << ",\"page_count\":" << image.page_count
              << ",\"page_channels\":["
              << image.pages[0].intermediate_count << ','
              << image.pages[1].intermediate_count << ','
              << image.pages[2].intermediate_count << ']'
              << ",\"baseline_gpu_ms\":" << baseline_ms
              << ",\"hybrid_gate_wall_ms\":" << gate_wall_ms
              << ",\"hybrid_down_wall_ms\":" << down_ms
              << ",\"hybrid_total_ms\":" << gate_wall_ms + down_ms
              << ",\"gpu_resident_gate_ms\":"
              << gate_lanes[0].interval.milliseconds()
              << ",\"cpu_resident_gate_ms\":"
              << gate_lanes[1].interval.milliseconds()
              << ",\"nvme_lane_gate_ms\":"
              << gate_lanes[2].interval.milliseconds()
              << ",\"nvme_direct_read_ms\":" << gate_lanes[2].io_ms
              << ",\"nvme_h2d_and_gate_ms\":" << gate_lanes[2].h2d_ms
              << ",\"triple_overlap_ms\":" << simultaneous_ms
              << ",\"nvme_direct_read_bytes\":"
              << gate_lanes[2].io_bytes
              << ",\"resident_gpu_weight_bytes\":"
              << resident_weight_bytes
              << ",\"paged_gpu_weight_high_water_bytes\":"
              << paged_vram_weight_bytes
              << ",\"resident_cpu_page_bytes\":"
              << image.pages[1].stored_bytes
              << ",\"nvme_staging_bytes\":" << image.pages[2].stored_bytes
              << ",\"source_fp4_bytes_exact\":true"
              << ",\"intermediate_max_abs_error\":"
              << intermediate_error.maximum_absolute
              << ",\"intermediate_cosine\":" << intermediate_error.cosine
              << ",\"output_max_abs_error\":"
              << output_error.maximum_absolute
              << ",\"output_relative_max_error\":"
              << output_error.relative_to_reference_maximum
              << ",\"output_rmse\":" << output_error.rmse
              << ",\"output_cosine\":" << output_error.cosine << "}\n";
    return valid ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "executable SwiGLU page experiment: " << error.what()
              << '\n';
    return 1;
  }
}
