if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  cmake_minimum_required(VERSION 3.24)
  project(expert_runtime LANGUAGES CXX)
endif()

option(EXPERT_RUNTIME_BUILD_TESTS "Build deterministic runtime tests" ON)
option(EXPERT_RUNTIME_ENABLE_CUDA_PINNED "Use cudaHostAlloc for the fixed host pool" OFF)
option(EXPERT_RUNTIME_ENABLE_CUDA_COMPUTE "Build SM86 Expert Pack kernels" OFF)

add_library(expert_runtime STATIC
  src/active_expert_executor.cpp
  src/adaptive_placement.cpp
  src/buffer_pool.cpp
  src/cache_state.cpp
  src/cpu/expert_executor.cpp
  src/cpu/fp4_host_executor.cpp
  src/expert_cache.cpp
  src/expert_catalog.cpp
  src/execution_provider.cpp
  src/expert_record.cpp
  src/expert_store.cpp
  src/gather_storage.cpp
  src/hybrid_dispatch.cpp
  src/model_artifact.cpp
  src/model_descriptor.cpp
  src/model_tensor_store.cpp
  src/moe_virtual_machine.cpp
  src/placement_profile.cpp
  src/program_executor.cpp
  src/resident_expert_set.cpp
  src/resource_governor.cpp
  src/route_census.cpp
  src/routed_expert_runtime.cpp
  src/scheduler.cpp
  src/sha256.cpp
  src/speculative_sampling.cpp
  src/telemetry.cpp
  src/worker_contract.cpp
  src/worker_provider.cpp
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
  target_link_libraries(expert_runtime PUBLIC bcrypt)
endif()
if(TARGET expert_core)
  target_link_libraries(expert_runtime PUBLIC expert_core)
endif()

add_executable(expert-fp16-cpu-value-profile
  tools/fp16_cpu_value_profile.cpp)
target_compile_features(expert-fp16-cpu-value-profile PRIVATE cxx_std_20)
if(MSVC)
  target_compile_options(expert-fp16-cpu-value-profile PRIVATE /arch:AVX2)
else()
  target_compile_options(expert-fp16-cpu-value-profile PRIVATE
    -mavx2 -mf16c -mfma)
endif()

if(MSVC)
  set_source_files_properties(
    src/cpu/expert_executor.cpp
    src/cpu/fp4_host_executor.cpp
    PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
  set_source_files_properties(
    src/cpu/expert_executor.cpp
    src/cpu/fp4_host_executor.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
endif()

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
    src/cuda/expert_directory.cu
    src/cuda/expert_uploader.cu
    src/cuda/flash_attention.cu
    src/cuda/moe_kernels.cu
    src/cuda/transformer_kernels.cu)
  target_compile_definitions(expert_runtime PUBLIC EXPERT_RUNTIME_HAS_CUDA_COMPUTE=1)
  if(MSVC)
    target_compile_options(expert_runtime PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
  endif()
  set_source_files_properties(src/cuda/flash_attention.cu PROPERTIES
    COMPILE_OPTIONS
      "--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math;-U__CUDA_NO_HALF_OPERATORS__;-U__CUDA_NO_HALF_CONVERSIONS__;-U__CUDA_NO_HALF2_OPERATORS__;-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
  target_include_directories(expert_runtime PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash_attention/shim
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash_attention/csrc/flash_attn/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash_attention/csrc/cutlass/include)
  target_link_libraries(expert_runtime PRIVATE CUDA::cudart CUDA::cublas)
  set_target_properties(expert_runtime PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  add_library(expert-active-expert-device STATIC
    src/cuda/active_expert_device_executor.cu)
  if(MSVC)
    target_compile_options(expert-active-expert-device PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:__cplusplus>)
  endif()
  target_link_libraries(expert-active-expert-device PUBLIC
    expert_runtime CUDA::cudart)
  set_target_properties(expert-active-expert-device PROPERTIES
    CUDA_ARCHITECTURES 86
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON)
  add_executable(expert-cuda-moe-smoke tools/cuda_moe_smoke.cpp)
  target_compile_features(expert-cuda-moe-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-cuda-moe-smoke PRIVATE expert_runtime CUDA::cudart)
  add_executable(expert-grouped-fp4-moe-cuda-smoke
    tools/grouped_fp4_moe_cuda_smoke.cpp)
  target_compile_features(expert-grouped-fp4-moe-cuda-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-grouped-fp4-moe-cuda-smoke PRIVATE
    expert_runtime CUDA::cudart)
  add_executable(expert-path-probe tools/expert_path_probe.cpp)
  target_compile_features(expert-path-probe PRIVATE cxx_std_20)
  target_link_libraries(expert-path-probe PRIVATE expert_runtime CUDA::cudart)
  if(MSVC)
    target_compile_options(expert-path-probe PRIVATE /arch:AVX2)
  else()
    target_compile_options(expert-path-probe PRIVATE -mavx2 -mfma)
  endif()
  add_executable(expert-qwen3-next-cuda-smoke tools/qwen3_next_cuda_smoke.cpp)
  target_compile_features(expert-qwen3-next-cuda-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-qwen3-next-cuda-smoke PRIVATE expert_runtime CUDA::cudart)
  add_executable(expert-dense-fp4-cuda-smoke
    tools/dense_fp4_cuda_smoke.cpp)
  target_compile_features(expert-dense-fp4-cuda-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-dense-fp4-cuda-smoke PRIVATE
    expert_runtime CUDA::cudart)
  add_executable(expert-mxfp6-cuda-smoke
    tools/mxfp6_cuda_smoke.cpp)
  target_compile_features(expert-mxfp6-cuda-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-mxfp6-cuda-smoke PRIVATE
    expert_runtime CUDA::cudart)
  add_executable(expert-native-nvfp4-cuda-smoke
    tools/native_nvfp4_cuda_smoke.cpp)
  target_compile_features(expert-native-nvfp4-cuda-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-native-nvfp4-cuda-smoke PRIVATE
    expert_runtime CUDA::cudart)
  add_library(expert-sm86-dense-moe-provider OBJECT tools/olmoe_runner.cpp)
  target_compile_features(expert-sm86-dense-moe-provider PRIVATE cxx_std_20)
  target_compile_definitions(expert-sm86-dense-moe-provider PRIVATE
    EXPERT_VM_PROVIDER_LIBRARY=1)
  target_link_libraries(expert-sm86-dense-moe-provider PRIVATE
    expert_runtime expert_core CUDA::cudart)
  add_library(expert-sm86-hybrid-delta-moe-provider OBJECT
    tools/qwen3_next_runner.cpp)
  target_compile_features(expert-sm86-hybrid-delta-moe-provider PRIVATE
    cxx_std_20)
  target_compile_definitions(expert-sm86-hybrid-delta-moe-provider PRIVATE
    EXPERT_VM_PROVIDER_LIBRARY=1)
  target_link_libraries(expert-sm86-hybrid-delta-moe-provider PRIVATE
    expert_runtime expert_core CUDA::cudart)
  add_library(expert-sm86-dense-fp4-provider OBJECT
    tools/dense_fp4_provider.cpp)
  target_compile_features(expert-sm86-dense-fp4-provider PRIVATE cxx_std_20)
  target_link_libraries(expert-sm86-dense-fp4-provider PRIVATE
    expert_runtime expert_core CUDA::cudart)
  add_executable(expert-moe-vm-runner
    tools/moe_vm_runner.cpp
    $<TARGET_OBJECTS:expert-sm86-dense-moe-provider>
    $<TARGET_OBJECTS:expert-sm86-hybrid-delta-moe-provider>
    $<TARGET_OBJECTS:expert-sm86-dense-fp4-provider>)
  target_compile_features(expert-moe-vm-runner PRIVATE cxx_std_20)
  target_link_libraries(expert-moe-vm-runner PRIVATE
    expert_runtime expert_core expert-active-expert-device CUDA::cudart)
  add_executable(expert-callable-vm-smoke
    tools/callable_vm_smoke.cpp
    $<TARGET_OBJECTS:expert-sm86-dense-moe-provider>
    $<TARGET_OBJECTS:expert-sm86-hybrid-delta-moe-provider>)
  target_compile_features(expert-callable-vm-smoke PRIVATE cxx_std_20)
  target_link_libraries(expert-callable-vm-smoke PRIVATE
    expert_runtime expert_core expert-active-expert-device CUDA::cudart)
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
  if(TARGET expert-grouped-fp4-moe-cuda-smoke)
    add_test(NAME expert-grouped-fp4-moe-cuda-smoke
      COMMAND expert-grouped-fp4-moe-cuda-smoke)
  endif()
  if(TARGET expert-dense-fp4-cuda-smoke)
    add_test(NAME expert-dense-fp4-cuda-smoke
      COMMAND expert-dense-fp4-cuda-smoke)
  endif()
  if(TARGET expert-mxfp6-cuda-smoke)
    add_test(NAME expert-mxfp6-cuda-smoke
      COMMAND expert-mxfp6-cuda-smoke)
  endif()
  if(TARGET expert-native-nvfp4-cuda-smoke)
    add_test(NAME expert-native-nvfp4-cuda-smoke
      COMMAND expert-native-nvfp4-cuda-smoke)
  endif()
endif()
