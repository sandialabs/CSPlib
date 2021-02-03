include_directories(.)
include_directories(../../core)
include_directories(../../include)

set ( EXEC  "driver_gODE_Davis_Skodje.exe" )
set ( SRCS  "driver_gODE_Davis_Skodje.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

set ( EXEC  "driver_gODE_Davis_Skodje_K.exe" )
set ( SRCS  "driver_gODE_Davis_Skodje_K.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

SET(copy_EXE_FILES
    driver_gODE_Davis_Skodje.exe
    driver_gODE_Davis_Skodje_K.exe
   )

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION example/kernel_class
)
