add_executable(expert-core-contract-tests core_contract_tests.cpp)
target_link_libraries(expert-core-contract-tests PRIVATE expert_core)
expert_set_warnings(expert-core-contract-tests)
add_test(NAME expert-core-contract COMMAND expert-core-contract-tests)

find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(
    NAME expert-feasibility-p1-cross-test
    COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/feasibility_cli_cross_test.py"
            "$<TARGET_FILE:expert-feasibility>"
            "${CMAKE_SOURCE_DIR}")
