# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find jemalloc: https://github.com/jemalloc/jemalloc
# If not in one of the default paths specify -D JEMALLOC_ROOT=/path/to/jemalloc
# to search there as well.

if(NOT JEMALLOC_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(JEMALLOC_ROOT "")
  set(JEMALLOC_ROOT $ENV{JEMALLOC_ROOT})
endif()

# find the jemalloc include directory
find_path(JEMALLOC_INCLUDE_DIRS jemalloc/jemalloc.h
  PATH_SUFFIXES include
  HINTS ${JEMALLOC_ROOT})

find_library(JEMALLOC_LIBRARIES
  NAMES jemalloc
  PATH_SUFFIXES lib64 lib
  HINTS ${JEMALLOC_ROOT})

set(JEMALLOC_VERSION "")

if(EXISTS "${JEMALLOC_INCLUDE_DIRS}/jemalloc/jemalloc.h")
  # Extract version info from header
  file(READ
    "${JEMALLOC_INCLUDE_DIRS}/jemalloc/jemalloc.h"
    JEMALLOC_FIND_HEADER_CONTENTS)

  string(REGEX MATCH "#define JEMALLOC_VERSION_MAJOR [0-9]+"
    JEMALLOC_MAJOR_VERSION "${JEMALLOC_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define JEMALLOC_VERSION_MAJOR " ""
    JEMALLOC_MAJOR_VERSION
    "${JEMALLOC_MAJOR_VERSION}")

  string(REGEX MATCH "#define JEMALLOC_VERSION_MINOR [0-9]+"
    JEMALLOC_MINOR_VERSION "${JEMALLOC_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define JEMALLOC_VERSION_MINOR " ""
    JEMALLOC_MINOR_VERSION
    "${JEMALLOC_MINOR_VERSION}")

  string(REGEX MATCH "#define JEMALLOC_VERSION_BUGFIX [0-9]+"
    JEMALLOC_SUBMINOR_VERSION "${JEMALLOC_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define JEMALLOC_VERSION_BUGFIX " ""
    JEMALLOC_SUBMINOR_VERSION
    "${JEMALLOC_SUBMINOR_VERSION}")

  set(JEMALLOC_VERSION
    "${JEMALLOC_MAJOR_VERSION}.${JEMALLOC_MINOR_VERSION}\
.${JEMALLOC_SUBMINOR_VERSION}"
    )
else()
  message(WARNING "Failed to find file "
    "'${JEMALLOC_INCLUDE_DIRS}/jemalloc/jemalloc.h' "
    "while detecting the JEMALLOC version.")
endif(EXISTS "${JEMALLOC_INCLUDE_DIRS}/jemalloc/jemalloc.h")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  JEMALLOC
  FOUND_VAR JEMALLOC_FOUND
  REQUIRED_VARS JEMALLOC_INCLUDE_DIRS JEMALLOC_LIBRARIES
  VERSION_VAR JEMALLOC_VERSION
  )
mark_as_advanced(JEMALLOC_INCLUDE_DIRS JEMALLOC_LIBRARIES
  JEMALLOC_MAJOR_VERSION JEMALLOC_MINOR_VERSION JEMALLOC_SUBMINOR_VERSION
  JEMALLOC_VERSION)
