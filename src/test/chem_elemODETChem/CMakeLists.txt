
include_directories(.)
include_directories(../../core)
include_directories(../../include)



set ( EXEC "driver_model_ODE_TChem2" )
set ( SRCS "driver_model_ODE_TChem2.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

SET(copy_EXE_FILES
    driver_model_ODE_TChem2
   )

SET(copy_OTHER_FILES
   )

INSTALL(FILES ${copy_OTHER_FILES}
        PERMISSIONS OWNER_READ OWNER_WRITE
        DESTINATION test/model_class
)

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION test/model_class
)
