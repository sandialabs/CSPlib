SET ( EXEC "eigentest.exe" )
SET ( SRCS "main-eigentest.cpp" )

IF (CSP_ENABLE_TPL_MAGMA)
  ADD_DEFINITIONS(-DTEST_MAGMA)
ENDIF()
IF (CSP_ENABLE_TPL_MKL)
  ADD_DEFINITIONS(-DTEST_MKL)
ENDIF()

ADD_EXECUTABLE( ${EXEC}  ${SRCS} )
TARGET_LINK_LIBRARIES( ${EXEC} ${CSP_LINK_LIBRARIES} )

SET(copy_EXE_FILES
    eigentest.exe
   )                 
                     
INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION example/eigentest
       )

