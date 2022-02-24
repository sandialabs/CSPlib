# Check TCHEM installation
FIND_PATH(TCHEM_FOUND_INCLUDE_PATH TC_interface.h HINTS ${TCHEM_INSTALL_PATH}/include)
FIND_LIBRARY(TCHEM_FOUND_LIBRARY NAMES tchem HINTS ${TCHEM_INSTALL_PATH}/lib)

IF (TCHEM_FOUND_INCLUDE_PATH AND TCHEM_FOUND_LIBRARY)
  ADD_LIBRARY(tchem UNKNOWN IMPORTED)
  SET_TARGET_PROPERTIES(tchem PROPERTIES 
    IMPORTED_LOCATION ${TCHEM_FOUND_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${TCHEM_FOUND_INCLUDE_PATH}
    INTERFACE_COMPILE_OPTIONS "-I${TCHEM_FOUND_INCLUDE_PATH}"
# Link option is available from cmake 3.14
#    INTERFACE_LINK_OPTIONS ""
    INTERFACE_LINK_LIBRARIES "-L${TCHEM_INSTALL_PATH}/lib -ltchem -ltchemutil")
  SET(TCHEM_FOUND ON)
ELSE()
  MESSAGE(FATAL_ERROR "-- TChem is not found at ${TCHEM_INSTALL_PATH}")
ENDIF()