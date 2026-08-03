#include "expert/runtime/cache_state.hpp"

namespace expert::runtime {

std::string_view cache_state_name(CacheState state) noexcept {
  switch (state) {
    case CacheState::absent:
      return "ABSENT";
    case CacheState::ssd_loading:
      return "SSD_LOADING";
    case CacheState::ram_ready:
      return "RAM_READY";
    case CacheState::gpu_uploading:
      return "GPU_UPLOADING";
    case CacheState::vram_ready:
      return "VRAM_READY";
    case CacheState::failed:
      return "FAILED";
  }
  return "UNKNOWN";
}

bool valid_cache_transition(CacheState from, CacheState to) noexcept {
  if (to == CacheState::failed) {
    return from != CacheState::failed;
  }
  switch (from) {
    case CacheState::absent:
      return to == CacheState::ssd_loading;
    case CacheState::ssd_loading:
      return to == CacheState::ram_ready || to == CacheState::absent;
    case CacheState::ram_ready:
      return to == CacheState::gpu_uploading || to == CacheState::absent;
    case CacheState::gpu_uploading:
      return to == CacheState::vram_ready || to == CacheState::ram_ready;
    case CacheState::vram_ready:
      return to == CacheState::ram_ready || to == CacheState::absent;
    case CacheState::failed:
      return to == CacheState::absent;
  }
  return false;
}

}  // namespace expert::runtime

