# Empty list
SET(CSPlib_SOURCES "")
SET(CSPlib_HEADERS "")

LIST(APPEND CSPlib_SOURCES
    tools.cpp
    model.cpp
    gODE.cpp
    kernel.cpp
    index.cpp
    )

LIST(APPEND CSPlib_HEADERS
            util.hpp tools.hpp
            model.hpp 
            kernel.hpp index.hpp
            CSPlib_CommandLineParser.hpp
            eigendecomposition_kokkos.hpp
            tools_tines.hpp
    )

## tchem interface
IF (CSPlib_ENABLE_TCHEMPP)

LIST(APPEND CSPlib_SOURCES
            chem_elem_ODE_TChem.cpp
    )

LIST(APPEND CSPlib_HEADERS
            chem_elem_ODE_TChem.hpp
    )

ENDIF()

ADD_LIBRARY(CSPlib ${CSPlib_SOURCES})

TARGET_LINK_LIBRARIES(${CSPlib_LINK_LIBRARIES})

##interface_TChem.hpp interface_TChem.cpp

INSTALL(FILES ${CSPlib_HEADERS} DESTINATION include)
