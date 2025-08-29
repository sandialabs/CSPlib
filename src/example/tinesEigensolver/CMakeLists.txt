include_directories(.)
include_directories(../../core)
include_directories(../../include)

set ( EXEC "runTinesEigensolver.exe" )
set ( SRCS "runTinesEigensolver.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

SET(copy_EXE_FILES runTinesEigensolver.exe)

SET(copy_OTHER_FILES)

INSTALL(FILES ${copy_OTHER_FILES}
        PERMISSIONS OWNER_READ OWNER_WRITE
        DESTINATION example/runTinesEigensolver)

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION example/runTinesEigensolver)
