# Empty list
SET(CSPlib_SOURCES "")
SET(CSPlib_HEADERS "")

LIST(APPEND CSPlib_SOURCES
    tools.cpp
    model.cpp
    gODE.cpp
    chem_elem_DAE.cpp
    kernel.cpp
    index.cpp
    indexBatch.cpp
    kernelBatch.cpp
    )

LIST(APPEND CSPlib_HEADERS
            util.hpp tools.hpp
            model.hpp chem_elem_DAE.hpp
            kernel.hpp index.hpp
            CSPlib_CommandLineParser.hpp
            eigendecomposition_kokkos.hpp
            index_kokkos.hpp
            tools_tines.hpp
            indexBatch.hpp
            kernelBatch.hpp
    )

## tchem interface
IF (CSPlib_ENABLE_TCHEMPP)

LIST(APPEND CSPlib_SOURCES
            chem_elem_ODE_TChem.cpp
            chem_elem_TCSTR_TChem.cpp
            chem_elem_TCSTRI_TChem.cpp
            chem_elem_DAE_PFR_TChem.cpp)

LIST(APPEND CSPlib_HEADERS
            chem_elem_ODE_TChem.hpp
            chem_elem_TCSTR_TChem.hpp
            chem_elem_TCSTRI_TChem.hpp
            chem_elem_DAE_PFR_TChem.hpp)

ENDIF()

ADD_LIBRARY(CSPlib ${CSPlib_SOURCES})

TARGET_LINK_LIBRARIES(${CSPlib_LINK_LIBRARIES})

INSTALL(FILES ${CSPlib_HEADERS} DESTINATION include)
