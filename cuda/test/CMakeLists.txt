find_package(GTest)

### - unit test
if(GTest_FOUND AND CUDA_FOUND)


    CUDA_add_executable(
        test_grad1conv
        ${CMAKE_CURRENT_SOURCE_DIR}/test_grad1conv.cu
    )

    target_link_libraries(
        test_grad1conv
        radial_kernels_grad1conv
        gtest
        pthread
    )

    #--------------------------------------------------------
    Add_library(
        testic OBJECT 
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu_conv.cxx
    )

    target_compile_options(
        testic BEFORE
        PRIVATE -include formula.h
    )

    Add_executable(
        test_conv
        $<TARGET_OBJECTS:testic>
    )

    target_link_libraries(
        test_conv
        shared_obj
        gtest
        pthread
    )

    set_target_properties( # pass needed options to add_executable()
        test_conv PROPERTIES
        LINKER_LANGUAGE CXX
    )

else()
    message(STATUS "   ==> No unit tests can be built.")
endif()

