if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(expert_runtime LANGUAGES CXX)
endif()

option(EXPERT_RUNTIME_BUILD_TESTS "Build deterministic runtime tests" ON)
option(EXPERT_RUNTIME_ENABLE_CUDA_PINNED "Use cudaHostAlloc for the fixed host pool" OFF)
option(EXPERT_RUNTIME_ENABLE_CUDA_COMPUTE "Build SM86 Expert Pack kernels" OFF)

add_library(expert_runtime STATIC
  src/buffer_pool.cpp
  src/cache_state.cpp
  src/expert_cache.cpp
  src/expert_record.cpp
  src/scheduler.cpp
  src/sha256.cpp
  src/telemetry.cpp
)

target_compile_features(expert_runtime PUBLIC cxx_std_20)
target_include_directories(expert_runtime PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
  target_compile_options(expert_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/EHsc>)
else()
  target_compile_options(expert_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>)
endif()

find_package(Threads REQUIRED)
target_link_libraries(expert_runtime PUBLIC Threads::Threads)

if(WIN32)
  target_sources(expert_runtime PRIVATE src/windows_iocp_storage.cpp)
endif()

if(EXPERT_RUNTIME_ENABLE_CUDA_PINNED)
  find_package(CUDAToolkit 12.1 REQUIRED)
  target_sources(expert_runtime PRIVATE src/cuda_pinned_allocator.cpp)
  target_compile_definitions(expert_runtime PUBLIC EXPERT_RUNTIME_HAS_CUDA_PINNED=1)
  target_link_libraries(expert_runtime PRIVATE CUDA::cudart)
endif()

if(EXPERT_RUNTIME_ENABLE_CUDA_COMPUTE)
  enable_language(CUDA)
  find_package(CUDAToolkit 12.1 REQUIRED)
  target_sources(expert_runtime PRIVATE
    src/cuda/expert_uploader.cu
    src/cuda/moe_kernels.cu
    src/cuda/transformer_kernels.cu)
  target_compile_definitions(expert_runtime PUBLIC EXPERT_RUNTIME_HAS_CUDA_COMPUTE=1)
  target_compile_definitions(expert_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH>)
  target_compile_options(expert_runtime PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>)
  target_link_libraries(expert_runtime PRIVATE CUDA::cudart)
  set_target_properties(expert_runtime PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  add_executable(expert-cuda-moe-smoke tools/cuda_moe_smoke.cpp)
  target_compile_features(expert-cuda-moe-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-cuda-moe-smoke PRIVATE expert_runtime CUDA::cudart)
  add_executable(expert-qwen3-next-cuda-smoke tools/qwen3_next_cuda_smoke.cpp)
  target_compile_features(expert-qwen3-next-cuda-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-qwen3-next-cuda-smoke PRIVATE expert_runtime CUDA::cudart)
  add_executable(expert-qwen3-next-runner tools/qwen3_next_runner.cpp)
  target_compile_features(expert-qwen3-next-runner PRIVATE cxx_std_20)
  target_link_libraries(expert-qwen3-next-runner PRIVATE expert_runtime expert_core CUDA::cudart)
  add_executable(expert-olmoe-runner tools/olmoe_runner.cpp)
  target_compile_features(expert-olmoe-runner PRIVATE cxx_std_20)
  target_link_libraries(expert-olmoe-runner PRIVATE expert_runtime expert_core CUDA::cudart)
endif()

if(EXPERT_RUNTIME_BUILD_TESTS)
  enable_testing()
  add_executable(expert_runtime_tests
    tests/cache_state_test.cpp
  )
  target_compile_features(expert_runtime_tests PRIVATE cxx_std_20)
  target_link_libraries(expert_runtime_tests PRIVATE expert_runtime)
  add_test(NAME expert_runtime_tests COMMAND expert_runtime_tests)
  if(TARGET expert-qwen3-next-cuda-smoke)
    add_test(NAME expert-qwen3-next-cuda-smoke
      COMMAND expert-qwen3-next-cuda-smoke)
  endif()
endif()
