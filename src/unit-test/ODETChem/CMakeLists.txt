include_directories(.)
include_directories(../../core)
include_directories(../../include)
# test
#
ADD_EXECUTABLE(csplib-test-odetchem.x CSPlib_Test_Driver_ODETChem.cpp)
TARGET_LINK_LIBRARIES(csplib-test-odetchem.x ${CSPlib_LINK_LIBRARIES} )

INSTALL(DIRECTORY inputs
        DESTINATION ${CMAKE_INSTALL_PREFIX}/unit-test/ODETChem)

INSTALL(DIRECTORY outputs_ref
        DESTINATION ${CMAKE_INSTALL_PREFIX}/unit-test/ODETChem)

FILE(COPY inputs
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

FILE(COPY outputs_ref
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

INSTALL(TARGETS csplib-test-odetchem.x
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION "${CMAKE_INSTALL_PREFIX}/${CSPlib_INSTALL_TEST_PATH}/ODETChem")

ADD_TEST(CSlibTest csplib-test-odetchem.x)
