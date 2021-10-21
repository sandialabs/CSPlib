include_directories(.)
include_directories(../../core)
include_directories(../../include)

set ( EXEC "run_index_ODE_TChem.exe" )
set ( SRCS "run_index_ODE_TChem.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

set ( EXEC "run_ODE_TChem_EigenK.exe" )
set ( SRCS "run_ODE_TChem_EigenK.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

set ( EXEC "runODETChemK.exe" )
set ( SRCS "runODETChemK.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )
SET(copy_EXE_FILES
    run_index_ODE_TChem.exe
    run_ODE_TChem_EigenK.exe
    runODETChemK.exe)

SET(copy_OTHER_FILES)

INSTALL(FILES ${copy_OTHER_FILES}
        PERMISSIONS OWNER_READ OWNER_WRITE
        DESTINATION example/indexODETChem
       )

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION example/indexODETChem)
