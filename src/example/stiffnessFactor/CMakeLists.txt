include_directories(.)
include_directories(../../core)
include_directories(../../include)

set ( EXEC  "DriverStiffnessFactor.exe" )
set ( SRCS  "DriverStiffnessFactor.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

SET(copy_EXE_FILES
   DriverStiffnessFactor.exe
    )

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION example/stiffness_factor
)
