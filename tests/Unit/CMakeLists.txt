# Distributed under the MIT License.
# See LICENSE.txt for details.

set(executable RunTests)

set(SPECTRE_TESTS)

set_property(GLOBAL PROPERTY SPECTRE_TESTS_LIB_FUNCTIONS_PROPERTY "")
set_property(GLOBAL PROPERTY SPECTRE_TESTS_LIBS_PROPERTY "")

add_subdirectory(ApparentHorizons)
add_subdirectory(ControlSystem)
add_subdirectory(DataStructures)
add_subdirectory(Domain)
add_subdirectory(Elliptic)
add_subdirectory(ErrorHandling)
add_subdirectory(Evolution)
add_subdirectory(Framework)
add_subdirectory(Helpers)
add_subdirectory(IO)
add_subdirectory(Informer)
add_subdirectory(NumericalAlgorithms)
add_subdirectory(Options)
add_subdirectory(Parallel)
add_subdirectory(ParallelAlgorithms)
add_subdirectory(PointwiseFunctions)
add_subdirectory(Time)
add_subdirectory(Utilities)
add_subdirectory(Visualization)

add_charm_module(${executable})

write_test_registration_function(
  "${CMAKE_BINARY_DIR}/tests/Unit/RunTestsRegister.hpp"
  "register_run_tests_libs"
  )

get_property(
  SPECTRE_TESTS_LIBRARIES
  GLOBAL
  PROPERTY
  SPECTRE_TESTS_LIBS_PROPERTY)

add_spectre_executable(
  ${executable}
  ${executable}.cpp
  ${executable}.decl.h
  ${executable}.def.h
  ${SPECTRE_TESTS}
  )

target_link_libraries(
  ${executable}
  PRIVATE
  ${SPECTRE_TESTS_LIBRARIES}
  Boost::boost
  # Link against Boost::program_options for now until we have proper
  # dependency handling for header-only libs
  Boost::program_options
  ErrorHandling
  Framework
  Informer
  Options
  SystemUtilities
  Utilities
  )

add_dependencies(
  ${executable}
  module_GlobalCache
  module_Main
  )

add_dependencies(unit-tests ${executable})

spectre_add_catch_tests(${executable} "${SPECTRE_TESTS_LIBRARIES}")

# Setup code coverage for unit tests
if(COVERAGE)
  # Setup unit test coverage target.
  setup_target_for_coverage(
      Unit
      ${CMAKE_BINARY_DIR}/docs/html
      unit-test-coverage
      ${CMAKE_CTEST_COMMAND}
      DEPENDS ${executable}
      TESTRUNNER_ARGS
      "--output-on-failure"
      IGNORE_COV
      '${CMAKE_BINARY_DIR}/Informer/InfoAtLink.cpp'
      '${CMAKE_SOURCE_DIR}/tests/*'
  )
endif()

add_subdirectory(RunSingleTest)

option(
  UNIT_TESTS_IN_TEST_EXECUTABLES
  "Build unit-tests as part of test-executables"
  ON
  )

if (UNIT_TESTS_IN_TEST_EXECUTABLES)
  add_dependencies(test-executables unit-tests)
endif (UNIT_TESTS_IN_TEST_EXECUTABLES)
