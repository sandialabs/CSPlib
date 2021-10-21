include_directories(.)
include_directories(../../core)
include_directories(../../include)

set ( EXEC "index_kokkos.exe" )
set ( SRCS "index_kokkos.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

set ( EXEC "index_kernel_batch.exe" )
set ( SRCS "index_kernel_batch.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

set ( EXEC "index_batch.exe" )
set ( SRCS "index_batch.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )

SET(copy_EXE_FILES
    index_kokkos.exe
    index_kernel_batch.exe
    index_batch.exe
   )

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION test/index_class_kokkos
)
