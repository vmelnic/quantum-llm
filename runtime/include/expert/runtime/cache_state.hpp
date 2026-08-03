#pragma once

#include <cstdint>
#include <string_view>

namespace expert::runtime {

enum class CacheState : std::uint8_t {
  absent,
  ssd_loading,
  ram_ready,
  gpu_uploading,
  vram_ready,
  failed,
};

[[nodiscard]] std::string_view cache_state_name(CacheState state) noexcept;

// Centralized validation keeps platform backends from publishing an expert by
// skipping checksum/upload completion states.
[[nodiscard]] bool valid_cache_transition(CacheState from,
                                          CacheState to) noexcept;

}  // namespace expert::runtime

