# Distributed under the MIT License.
# See LICENSE.txt for details.

add_subdirectory(Limiters)

set(LIBRARY "Test_EvolutionDg")

set(LIBRARY_SOURCES
  Actions/Test_ApplyBoundaryCorrections.cpp
  Actions/Test_ComputeTimeDerivative.cpp
  Actions/Test_NormalCovectorAndMagnitude.cpp
  Initialization/Test_Mortars.cpp
  Initialization/Test_QuadratureTag.cpp
  Test_BoundaryCorrectionsHelper.cpp
  Test_InboxTags.cpp
  Test_LiftFromBoundary.cpp
  Test_MortarData.cpp
  Test_MortarTags.cpp
  Test_NormalVectorTags.cpp
  Test_ProjectToBoundary.cpp
  )

add_test_library(
  ${LIBRARY}
  "Evolution/DiscontinuousGalerkin/"
  "${LIBRARY_SOURCES}"
  "Boost::boost;Evolution;EvolutionDgActionsHelpers"
  )

add_dependencies(
  ${LIBRARY}
  module_GlobalCache
  )
