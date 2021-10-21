include_directories(.)
include_directories(../../core)
include_directories(../../include)

set ( EXEC "kernel_kokkos.exe" )
set ( SRCS "kernel_kokkos.cpp" )

add_executable( ${EXEC}  ${SRCS} )
target_link_libraries( ${EXEC} ${CSPlib_LINK_LIBRARIES} )


SET(copy_EXE_FILES
    kernel_kokkos.exe
   )

INSTALL(TARGETS ${copy_EXE_FILES}
        PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE
        DESTINATION test/kernel_kokkos
)
