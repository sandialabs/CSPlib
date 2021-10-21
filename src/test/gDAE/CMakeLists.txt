
include_directories(.)     
include_directories(../../core)     
include_directories(../../include)

set ( EXEC "driver_gDAE" )
set ( SRCS "driver_gDAE.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

SET(copy_EXE_FILES
    driver_gDAE
   )                 

INSTALL(FILES ${copy_OTHER_FILES}
        PERMISSIONS OWNER_READ OWNER_WRITE
        DESTINATION test/model_class
)

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION test/model_class
)

