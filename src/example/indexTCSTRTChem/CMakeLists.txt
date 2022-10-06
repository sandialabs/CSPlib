include_directories(.)
include_directories(../../core)
include_directories(../../include)

set ( EXEC "run_index_TCSTR_TChem.exe" )
set ( SRCS "run_index_TCSTR_TChem.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

set ( EXEC "runTCSTRI_TChem.exe" )
set ( SRCS "runTCSTRI_TChem.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

SET(copy_EXE_FILES
    run_index_TCSTR_TChem.exe
    runTCSTRI_TChem.exe
   )

SET(copy_OTHER_FILES
    #chem.inp
    #therm.dat
    #periodictable.dat
   )

INSTALL(FILES ${copy_OTHER_FILES}
        PERMISSIONS OWNER_READ OWNER_WRITE
        DESTINATION example/indexTCSTRTChem
       )

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION example/indexTCSTRTChem
       )
