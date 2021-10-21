# Kokkos requires 3.3
CMAKE_MINIMUM_REQUIRED(VERSION 3.3)

# Project CSPlib
PROJECT (CSPlib)

# CMAKE setup
SET(CMAKE_VERBOSE_MAKEFILE ON) # see all output
SET(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR})
SET(CMAKE_CXX_STANDARD 11)

# Define BUILD_TESTING in cmake
INCLUDE(CTest)

# Installation
IF (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  SET(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}" CACHE PATH "FOO install prefix" FORCE)
ENDIF()

SET(INSTALL_LIB_DIR     lib    )
SET(INSTALL_BIN_DIR     bin    )
SET(INSTALL_INCLUDE_DIR include)
SET(CSPlib_INSTALL_TEST_PATH    unit-test)

# CSPlib link libraries for tests (tchem and CSPlib are required)
SET(CSPlib_LINK_LIBRARIES CSPlib)

#
# Options : use CSPlib prefix
#
OPTION(CSPLIB_MESUARE_WALL_TIME "Flag to save wall times" OFF)
OPTION(CSPLIB_ENABLE_TEST "Flag to enable unit tests" OFF)
OPTION(TCHEM_INSTALL_PATH "Path to TChem installation")
OPTION(KOKKOS_INSTALL_PATH "Path to Kokkos installation")
OPTION(TINES_INSTALL_PATH "Path to Tines installation")
OPTION(GTEST_INSTALL_PATH "Path to gtest installation")
SET(HAVE_KOKKOS ${CSPlib_ENABLE_KOKKOS})

# Define configure-related preprocessor macros
INCLUDE_DIRECTORIES(${CSPlib_BINARY_DIR})


#
# Kokkos
#
IF (KOKKOS_INSTALL_PATH)
  # Interface installed Kokkos
  MESSAGE("-- CSPlib uses Kokkos installed at ${KOKKOS_INSTALL_PATH}")
  FIND_PACKAGE(Kokkos REQUIRED)
  SET(CSPlib_INTERNAL_KOKKOS_TARGET "Kokkos::kokkoscore;Kokkos::kokkoscontainers")
  SET(CSPlib_ENABLE_TPL_KOKKOS ON CACHE BOOL "Enable Kokkos")
  LIST(APPEND CSPlib_LINK_LIBRARIES ${CSPlib_INTERNAL_KOKKOS_TARGET})
ELSE()
  # Process submodule Kokkos (includes, cxx flags and lib)
  MESSAGE(FATAL_ERROR "-- KOKKOS_INSTALL_PATH is not provided")
ENDIF()



#
# gtest
#
IF (CSPLIB_ENABLE_TEST)
  IF (GTEST_INSTALL_PATH)
    # Interface installed gtest
    MESSAGE("-- CSPlib uses gtest installed at ${GTEST_INSTALL_PATH}")
    FIND_PACKAGE(Gtest REQUIRED)
    SET(CSPlib_INTERNAL_TINES_TARGET "GTest::gtest_main")
  ELSE()
    MESSAGE(FATAL_ERROR "-- Test is enabled but Gtest install path is not provided")
  ENDIF()
  LIST(APPEND CSPlib_LINK_LIBRARIES ${CSPlib_INTERNAL_TINES_TARGET})
ENDIF()

#
# Tines
#
IF (TINES_INSTALL_PATH)
  MESSAGE(STATUS "CSPlib attempts to use Tines installed at ${TINES_INSTALL_PATH}")
  FIND_PACKAGE(Tines REQUIRED)
  SET(CSPlib_INTERNAL_TINES_TARGET "Tines::tines")
  SET(CSPlib_ENABLE_TPL_TINES ON CACHE BOOL "Enable Tines")
  LIST(APPEND CSPlib_LINK_LIBRARIES ${CSPlib_INTERNAL_TINES_TARGET})
ELSE()
  MESSAGE(FATAL_ERROR "-- Tines install path is not provided")
ENDIF()

#
# TChem
#
IF (TCHEM_INSTALL_PATH)
  SET(CSPlib_ENABLE_TCHEMPP ON)
  FIND_PACKAGE(TCHEMPP)
  IF (TCHEM_FOUND)
    SET(CSPlib_INTERNAL_TCHEM_TARGET "TChem::tchem")
    SET(CSPlib_ENABLE_TPL_TCHEM ON CACHE BOOL "Enable TChem")
    LIST(APPEND CSPlib_LINK_LIBRARIES ${CSPlib_INTERNAL_TCHEM_TARGET})
    MESSAGE("-- TChem is requested and found")
  ENDIF()
  IF (TCHEM_FOUND)
  ELSE()
    MESSAGE(FATAL_ERROR "-- TChem is not found at ${TCHEM_INSTALL_PATH}")
  ENDIF()
ENDIF()

MESSAGE("-- CSPlib_LINK_LIBRARIES = ${CSPlib_LINK_LIBRARIES}")

CONFIGURE_FILE (
  ${CSPlib_SOURCE_DIR}/include/CSPlib_ConfigDefs.h.in
  ${CSPlib_BINARY_DIR}/CSPlib_ConfigDefs.h
  )
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/CSPlib_ConfigDefs.h
  DESTINATION "${CMAKE_INSTALL_PREFIX}/${CSPlib_INSTALL_INCLUDE_PATH}")

IF (CSPLIB_ENABLE_TEST)
    ADD_SUBDIRECTORY (unit-test)
ENDIF()

ADD_SUBDIRECTORY (core)
ADD_SUBDIRECTORY (test)
ADD_SUBDIRECTORY (example)
