#include "expert/runtime/model_tensor_store.hpp"

#include "expert/runtime/sha256.hpp"

#include <cerrno>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace expert::runtime {
namespace {

Status copy_status(const Status& status) {
  return {status.code(), std::string(status.message())};
}

class MappedPack final {
 public:
  MappedPack() = default;
  MappedPack(const MappedPack&) = delete;
  MappedPack& operator=(const MappedPack&) = delete;
  ~MappedPack() {
#ifdef _WIN32
    if (data_ != nullptr) UnmapViewOfFile(data_);
    if (mapping_ != nullptr) CloseHandle(mapping_);
    if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
    if (data_ != nullptr)
      munmap(const_cast<std::byte*>(data_), bytes_);
#endif
  }

  [[nodiscard]] static Status open(const ArtifactPack& artifact,
                                   std::shared_ptr<MappedPack>& output) {
    if (artifact.bytes == 0U ||
        artifact.bytes > std::numeric_limits<std::size_t>::max())
      return {ErrorCode::invalid_argument,
              "dense artifact pack exceeds the address space"};
    auto pack = std::make_shared<MappedPack>();
    pack->bytes_ = static_cast<std::size_t>(artifact.bytes);
#ifdef _WIN32
    pack->file_ = CreateFileW(
        artifact.path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
        nullptr);
    if (pack->file_ == INVALID_HANDLE_VALUE)
      return {ErrorCode::open_failed, "cannot open dense artifact pack"};
    pack->mapping_ = CreateFileMappingW(pack->file_, nullptr, PAGE_READONLY,
                                        0U, 0U, nullptr);
    if (pack->mapping_ == nullptr)
      return {ErrorCode::open_failed,
              "cannot create dense artifact file mapping"};
    pack->data_ = static_cast<const std::byte*>(
        MapViewOfFile(pack->mapping_, FILE_MAP_READ, 0U, 0U, 0U));
    if (pack->data_ == nullptr)
      return {ErrorCode::open_failed, "cannot map dense artifact pack"};
#else
    const auto descriptor = ::open(artifact.path.c_str(), O_RDONLY);
    if (descriptor < 0)
      return {ErrorCode::open_failed,
              std::string("cannot open dense artifact pack: ") +
                  std::strerror(errno)};
    void* mapped = mmap(nullptr, pack->bytes_, PROT_READ, MAP_PRIVATE,
                        descriptor, 0);
    const auto saved_errno = errno;
    static_cast<void>(::close(descriptor));
    if (mapped == MAP_FAILED)
      return {ErrorCode::open_failed,
              std::string("cannot map dense artifact pack: ") +
                  std::strerror(saved_errno)};
    pack->data_ = static_cast<const std::byte*>(mapped);
#endif
    if (!constant_time_equal(
            sha256({pack->data_, pack->bytes_}), artifact.sha256))
      return {ErrorCode::checksum_mismatch,
              "dense artifact pack checksum mismatch"};
    output = std::move(pack);
    return Status::success();
  }

  [[nodiscard]] const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

 private:
#ifdef _WIN32
  HANDLE file_{INVALID_HANDLE_VALUE};
  HANDLE mapping_{};
#endif
  const std::byte* data_{};
  std::size_t bytes_{};
};

}  // namespace

struct MappedModelTensorStore::Core final {
  std::map<std::string, std::shared_ptr<MappedPack>, std::less<>> packs;
  std::map<std::string, std::shared_ptr<const ImmutableModelTensor>,
           std::less<>>
      tensors;
};

MappedModelTensorStore::MappedModelTensorStore() = default;
MappedModelTensorStore::MappedModelTensorStore(
    std::shared_ptr<Core> core) noexcept
    : core_(std::move(core)) {}
MappedModelTensorStore::MappedModelTensorStore(
    MappedModelTensorStore&&) noexcept = default;
MappedModelTensorStore& MappedModelTensorStore::operator=(
    MappedModelTensorStore&&) noexcept = default;
MappedModelTensorStore::~MappedModelTensorStore() = default;

bool MappedModelTensorStore::valid() const noexcept { return core_ != nullptr; }

Status MappedModelTensorStore::create(
    const ModelArtifact& artifact,
    MappedModelTensorStore& destination) noexcept {
  try {
    if (destination.valid())
      return {ErrorCode::invalid_argument,
              "mapped tensor store destination is already initialized"};
    auto core = std::make_shared<Core>();
    for (const auto& pack : artifact.packs()) {
      if (pack.kind != "dense") continue;
      std::shared_ptr<MappedPack> mapped;
      const auto opened = MappedPack::open(pack, mapped);
      if (!opened.ok()) return copy_status(opened);
      if (!core->packs.emplace(pack.name, std::move(mapped)).second)
        return {ErrorCode::invalid_argument,
                "dense artifact pack name is duplicated"};
    }
    for (const auto& entry : artifact.dense_tensors()) {
      const auto pack = core->packs.find(entry.pack);
      if (pack == core->packs.end() || entry.stored_bytes == 0U ||
          entry.record_offset > pack->second->bytes() ||
          entry.stored_bytes >
              pack->second->bytes() - entry.record_offset ||
          entry.data_offset > entry.stored_bytes ||
          entry.data_bytes > entry.stored_bytes - entry.data_offset ||
          entry.scale_offset > entry.stored_bytes ||
          entry.scale_bytes > entry.stored_bytes - entry.scale_offset)
        return {ErrorCode::invalid_argument,
                "dense tensor record is outside its mapped pack"};
      auto tensor = std::make_shared<ImmutableModelTensor>();
      tensor->name = entry.name;
      tensor->encoding = entry.encoding;
      tensor->quant_abi = entry.quant_abi;
      tensor->shape = entry.shape;
      tensor->data_offset = entry.data_offset;
      tensor->data_bytes = entry.data_bytes;
      tensor->scale_offset = entry.scale_offset;
      tensor->scale_bytes = entry.scale_bytes;
      tensor->value = {
          "artifact.dense-record.v1", "host.mmap.readonly", pack->second,
          pack->second->data() + entry.record_offset, entry.stored_bytes};
      if (!core->tensors.emplace(tensor->name, std::move(tensor)).second)
        return {ErrorCode::invalid_argument,
                "dense artifact tensor name is duplicated"};
    }
    destination = MappedModelTensorStore(std::move(core));
    return Status::success();
  } catch (const std::exception& error) {
    return {ErrorCode::internal,
            std::string("mapped model tensor store failed: ") + error.what()};
  }
}

ResolveModelTensorResult MappedModelTensorStore::resolve(
    std::string_view name) {
  if (!core_)
    return {{ErrorCode::invalid_argument,
             "mapped model tensor store is not initialized"},
            {}};
  const auto found = core_->tensors.find(name);
  if (found == core_->tensors.end())
    return {{ErrorCode::invalid_argument,
             "artifact tensor is absent from the dense index"},
            {}};
  return {Status::success(), found->second};
}

}  // namespace expert::runtime
