find_package(GTest)

### - unit test
if(GTest_FOUND)

    Add_library(
        testic OBJECT 
        ${CMAKE_CURRENT_SOURCE_DIR}/unit/gpu_conv.cxx
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
    message(STATUS "   ==> No unit tests can be build.")
endif()

