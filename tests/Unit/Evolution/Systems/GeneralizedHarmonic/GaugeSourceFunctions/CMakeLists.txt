# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "Test_GeneralizedHarmonicGaugeSourceFunctions")

set(LIBRARY_SOURCES
  Test_DampedHarmonic.cpp
  Test_DampedWaveHelpers.cpp
  Test_InitializeDampedHarmonic.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/"
  "${LIBRARY_SOURCES}"
  "GeneralizedHarmonicGaugeSourceFunctions;Options"
  )
