enable_testing()

find_package(Threads)

function(add_test_helper test_name source_file)
    add_executable(${test_name} ${source_file})
    target_link_libraries(${test_name} gtest gtest_main)
    add_test(${test_name} ${test_name})
endfunction()

# Multi thread compilation test
add_test_helper(unit_test_compile gtest_compile.cpp)


if (USE_CUDA)
        # ------------------ grad1conv test
        CUDA_add_executable(
            unit_test_grad1conv
            ${CMAKE_CURRENT_SOURCE_DIR}/gtest_grad1conv.cu
        )

        target_link_libraries(
            unit_test_grad1conv
            radial_kernels_grad1conv
            gtest
            pthread
        )

        add_test(unit_test_conv unit_test_conv)

        # ------------------ conv test
        Add_library(
            testic OBJECT
            ${CMAKE_CURRENT_SOURCE_DIR}/gtest_conv.cpp
        )

        target_compile_options(
            testic BEFORE
            PRIVATE ${IncludePrefix} ${shared_obj_name}.h
        )

        Add_dependencies(
            testic
            gtest
        )

        Add_executable(
            unit_test_conv
            $<TARGET_OBJECTS:testic>
        )

        target_link_libraries(
            unit_test_conv
            ${shared_obj_name}
            gtest
            pthread
        )

        set_target_properties( # pass needed options to add_executable()
            unit_test_conv PROPERTIES
            LINKER_LANGUAGE CXX
        )

        add_test(unit_test_conv unit_test_conv)
endif()
