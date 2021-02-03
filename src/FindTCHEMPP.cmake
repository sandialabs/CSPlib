# Check TChem installation
FILE(GLOB_RECURSE TCHEM_FOUND_CMAKE_FILE "${TCHEM_INSTALL_PATH}/TChemConfig.cmake")

MESSAGE(STATUS "TChem install path : ${TCHEM_INSTALL_PATH}")
MESSAGE(STATUS "TChem found cmake config : ${TCHEM_FOUND_CMAKE_FILE}")

IF (TCHEM_FOUND_CMAKE_FILE)
  INCLUDE(${TCHEM_FOUND_CMAKE_FILE})
  SET(TCHEM_FOUND ON)
ELSE()
  MESSAGE(FATAL_ERROR "-- TChem is not found at ${TCHEM_INSTALL_PATH}")
ENDIF()