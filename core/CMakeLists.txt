add_library(expert_core STATIC
    src/feasibility.cpp
    src/json.cpp
    src/sha256.cpp)

target_include_directories(expert_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_features(expert_core PUBLIC cxx_std_20)
expert_set_warnings(expert_core)

add_executable(expert-feasibility tools/feasibility_main.cpp)
target_link_libraries(expert-feasibility PRIVATE expert_core)
expert_set_warnings(expert-feasibility)
