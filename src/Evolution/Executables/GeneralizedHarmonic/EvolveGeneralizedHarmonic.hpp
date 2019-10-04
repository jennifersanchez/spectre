// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <vector>

#include "AlgorithmSingleton.hpp"
#include "ApparentHorizons/ComputeItems.hpp"
#include "ApparentHorizons/Tags.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Evolution/Actions/ComputeTimeDerivative.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/DgElementArray.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Filtering.hpp"
#include "Evolution/DiscontinuousGalerkin/ObserveFields.hpp"
#include "Evolution/DiscontinuousGalerkin/ObserveNorms.hpp"
#include "Evolution/EventsAndTriggers/Actions/RunEventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Event.hpp"
#include "Evolution/EventsAndTriggers/EventsAndTriggers.hpp"  // IWYU pragma: keep
#include "Evolution/EventsAndTriggers/Tags.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Initialization/NonconservativeSystem.hpp"
#include "Evolution/NumericalInitialData.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/BoundaryConditions.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Equations.hpp"  // IWYU pragma: keep // for UpwindFlux
#include "Evolution/Systems/GeneralizedHarmonic/Initialize.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "IO/DataImporter/DataFileReader.hpp"
#include "IO/DataImporter/ElementActions.hpp"
#include "IO/Observer/Actions.hpp"  // IWYU pragma: keep
#include "IO/Observer/Helpers.hpp"
#include "IO/Observer/ObserverComponent.hpp"  // IWYU pragma: keep
#include "IO/Observer/RegisterObservers.hpp"
#include "IO/Observer/Tags.hpp"  // IWYU pragma: keep
#include "Informer/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyBoundaryFluxesLocalTimeStepping.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ApplyFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/AddTemporalIdsToInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/FindApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/Callbacks/ObserveTimeSeriesOnSurface.hpp"
#include "NumericalAlgorithms/Interpolation/CleanUpInterpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolate.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTarget.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetApparentHorizon.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolationTargetReceiveVars.hpp"
#include "NumericalAlgorithms/Interpolation/Interpolator.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceivePoints.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorReceiveVolumeData.hpp"
#include "NumericalAlgorithms/Interpolation/InterpolatorRegisterElement.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"
#include "NumericalAlgorithms/Interpolation/TryToInterpolate.hpp"
#include "Options/Options.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Reduction.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/RemoveOptionsAndTerminatePhase.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Time/Actions/AdvanceTime.hpp"            // IWYU pragma: keep
#include "Time/Actions/ChangeStepSize.hpp"         // IWYU pragma: keep
#include "Time/Actions/RecordTimeStepperData.hpp"  // IWYU pragma: keep
#include "Time/Actions/SelfStartActions.hpp"       // IWYU pragma: keep
#include "Time/Actions/UpdateU.hpp"                // IWYU pragma: keep
#include "Time/StepChoosers/Cfl.hpp"               // IWYU pragma: keep
#include "Time/StepChoosers/Constant.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/Increase.hpp"          // IWYU pragma: keep
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/StepControllers/StepController.hpp"
#include "Time/Tags.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Time/Triggers/TimeTriggers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
// IWYU pragma: no_forward_declare MathFunction
struct Inertial;
}  // namespace Frame
namespace Parallel {
template <typename Metavariables>
class CProxy_ConstGlobalCache;
}  // namespace Parallel
/// \endcond

struct EvolutionMetavars {
  // Customization/"input options" to simulation
  static constexpr int volume_dim = 3;
  using frame = Frame::Inertial;
  using system = GeneralizedHarmonic::System<volume_dim>;
  using temporal_id = Tags::TimeStepId;
  static constexpr bool local_time_stepping = false;

  using analytic_solution =
      GeneralizedHarmonic::Solutions::WrappedGr<gr::Solutions::KerrSchild>;
  using analytic_solution_tag = Tags::AnalyticSolution<analytic_solution>;
  using initial_data_tag = Tags::AnalyticSolution<analytic_solution>;
  using boundary_condition_tag = initial_data_tag;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, frame>,
                 gr::Tags::RicciTensor<volume_dim, frame, DataVector>>;

  // The type of initial data for the evolution. Set to `analytic_solution` for
  // starting from an analytic solution, or `NumericalInitialData` to read
  // data from the disk.
  using initial_data = NumericalInitialData<system>;

  using normal_dot_numerical_flux =
      Tags::NumericalFlux<GeneralizedHarmonic::UpwindFlux<volume_dim>>;

  using constraint_tags = tmpl::list<
      GeneralizedHarmonic::Tags::GaugeConstraint<volume_dim, frame>,
      GeneralizedHarmonic::Tags::TwoIndexConstraint<volume_dim, frame>,
      GeneralizedHarmonic::Tags::FConstraint<volume_dim, frame>,
      GeneralizedHarmonic::Tags::ThreeIndexConstraint<volume_dim, frame>,
      GeneralizedHarmonic::Tags::FourIndexConstraint<volume_dim, frame>,
      GeneralizedHarmonic::Tags::ConstraintEnergy<volume_dim, frame>>;

  // HACK until proper compute tags are cherry-picked
  struct Unity : db::ComputeTag {
    static std::string name() noexcept { return "Unity"; }
    static Scalar<DataVector> function(
        const Scalar<DataVector>& used_for_size) noexcept {
      return make_with_value<Scalar<DataVector>>(used_for_size, 1.0);
    }
    using argument_tags =
        tmpl::list<StrahlkorperGr::Tags::AreaElement<Frame::Inertial>>;
  };

  struct Horizon {
    using tags_to_observe =
        tmpl::list<StrahlkorperGr::Tags::SurfaceIntegral<Unity, frame>>;
    using compute_items_on_source = tmpl::list<
        gr::Tags::SpatialMetricCompute<volume_dim, frame, DataVector>,
        ah::Tags::InverseSpatialMetricCompute<volume_dim, frame>,
        ah::Tags::ExtrinsicCurvatureCompute<volume_dim, frame>,
        ah::Tags::SpatialChristoffelSecondKindCompute<volume_dim, frame>>;
    using vars_to_interpolate_to_target =
        tmpl::list<gr::Tags::SpatialMetric<volume_dim, frame, DataVector>,
                   gr::Tags::InverseSpatialMetric<volume_dim, frame>,
                   gr::Tags::ExtrinsicCurvature<volume_dim, frame>,
                   gr::Tags::SpatialChristoffelSecondKind<volume_dim, frame>,
                   gr::Tags::RicciTensor<volume_dim, frame, DataVector>>;
    using compute_items_on_target = tmpl::list<
        StrahlkorperGr::Tags::AreaElement<frame>, StrahlkorperGr::Tags::Unity,
        StrahlkorperGr::Tags::SurfaceIntegral<StrahlkorperGr::Tags::Unity,
                                              frame>>;
    using compute_target_points =
        intrp::Actions::ApparentHorizon<Horizon, ::Frame::Inertial>;
    using post_interpolation_callback =
        intrp::callbacks::FindApparentHorizon<Horizon>;
    using post_horizon_find_callback =
        intrp::callbacks::ObserveTimeSeriesOnSurface<tags_to_observe, Horizon,
                                                     Horizon>;
  };
  using interpolation_target_tags = tmpl::list<Horizon>;
  using interpolator_source_vars =
      tmpl::list<gr::Tags::SpacetimeMetric<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Pi<volume_dim, frame>,
                 GeneralizedHarmonic::Tags::Phi<volume_dim, frame>>;

  using observation_events = tmpl::list<
      dg::Events::Registrars::ObserveNorms<volume_dim, constraint_tags>,
      dg::Events::Registrars::ObserveFields<
          volume_dim,
          tmpl::append<
              typename system::variables_tag::tags_list,
              tmpl::list<
                  ::Tags::PointwiseL2Norm<
                      GeneralizedHarmonic::Tags::GaugeConstraint<volume_dim,
                                                                 frame>>,
                  ::Tags::PointwiseL2Norm<
                      GeneralizedHarmonic::Tags::TwoIndexConstraint<volume_dim,
                                                                    frame>>,
                  ::Tags::PointwiseL2Norm<
                      GeneralizedHarmonic::Tags::ThreeIndexConstraint<
                          volume_dim, frame>>,
                  ::Tags::PointwiseL2Norm<
                      GeneralizedHarmonic::Tags::FourIndexConstraint<
                          volume_dim, frame>>,
                  ::Tags::PointwiseL2Norm<GeneralizedHarmonic::Tags::
                                              FConstraint<volume_dim, frame>>>>,
          typename system::variables_tag::tags_list>>;
  using triggers = Triggers::time_triggers;

  // Events include the observation events and finding the horizon
  using events = tmpl::push_back<
      observation_events,
      intrp::Events::Registrars::Interpolate<3, interpolator_source_vars>>;

  // A tmpl::list of tags to be added to the ConstGlobalCache by the
  // metavariables
  using const_global_cache_tags = tmpl::list<
      initial_data_tag,
      Tags::TimeStepper<tmpl::conditional_t<local_time_stepping, LtsTimeStepper,
                                            TimeStepper>>,
      GeneralizedHarmonic::Tags::GaugeHRollOnStartTime,
      GeneralizedHarmonic::Tags::GaugeHRollOnTimeWindow,
      GeneralizedHarmonic::Tags::GaugeHSpatialWeightDecayWidth<frame>,
      Tags::EventsAndTriggers<events, triggers>>;

  using step_choosers =
      tmpl::list<StepChoosers::Registrars::Cfl<volume_dim, frame>,
                 StepChoosers::Registrars::Constant,
                 StepChoosers::Registrars::Increase>;

  struct ObservationType {};
  using element_observation_type = ObservationType;

  using observed_reduction_data_tags = observers::collect_reduction_data_tags<
      tmpl::push_back<Event<observation_events>::creatable_classes,
                      typename Horizon::post_horizon_find_callback>>;

  using compute_rhs = tmpl::flatten<tmpl::list<
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::InternalDirections<volume_dim>>,
      dg::Actions::SendDataForFluxes<EvolutionMetavars>,
      Actions::ComputeTimeDerivative,
      dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::BoundaryDirectionsInterior<volume_dim>>,
      dg::Actions::ReceiveDataForFluxes<EvolutionMetavars>,
      tmpl::conditional_t<local_time_stepping, tmpl::list<>,
                          dg::Actions::ApplyFluxes>,
      GeneralizedHarmonic::Actions::
          ImposeConstraintPreservingBoundaryConditions<EvolutionMetavars>,
      Actions::RecordTimeStepperData>>;
  using update_variables = tmpl::flatten<tmpl::list<
      tmpl::conditional_t<local_time_stepping,
                          dg::Actions::ApplyBoundaryFluxesLocalTimeStepping,
                          tmpl::list<>>,
      Actions::UpdateU  //,
                        // dg::Actions::ExponentialFilter<
      // 0, typename system::variables_tag::type::tags_list>
      >>;

  enum class Phase {
    Initialization,
    InitializeTimeStepperHistory,
    Register,
    ImportData,
    Evolve,
    Exit
  };

  using initialization_actions = tmpl::list<
      dg::Actions::InitializeDomain<volume_dim>,
      Initialization::Actions::NonconservativeSystem,
      GeneralizedHarmonic::Actions::InitializeGHAnd3Plus1VariablesTags<
          volume_dim>,
      dg::Actions::InitializeInterfaces<
          system,
          dg::Initialization::slice_tags_to_face<
              typename system::variables_tag,
              gr::Tags::SpatialMetricCompute<volume_dim, frame, DataVector>,
              gr::Tags::DetAndInverseSpatialMetricCompute<volume_dim, frame,
                                                          DataVector>,
              gr::Tags::ShiftCompute<volume_dim, frame, DataVector>,
              gr::Tags::LapseCompute<volume_dim, frame, DataVector>>,
          dg::Initialization::slice_tags_to_exterior<
              typename system::variables_tag,
              gr::Tags::SpatialMetricCompute<volume_dim, frame, DataVector>,
              gr::Tags::DetAndInverseSpatialMetricCompute<volume_dim, frame,
                                                          DataVector>,
              gr::Tags::ShiftCompute<volume_dim, frame, DataVector>,
              gr::Tags::LapseCompute<volume_dim, frame, DataVector>>,
          dg::Initialization::face_compute_tags<
              ::Tags::BoundaryCoordinates<volume_dim, frame>,
              GeneralizedHarmonic::Tags::ConstraintGamma0Compute<volume_dim,
                                                                 frame>,
              GeneralizedHarmonic::Tags::ConstraintGamma1Compute<volume_dim,
                                                                 frame>,
              GeneralizedHarmonic::Tags::ConstraintGamma2Compute<volume_dim,
                                                                 frame>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>,
              GeneralizedHarmonic::CharacteristicSpeedsCompute<volume_dim,
                                                               frame>>,
          dg::Initialization::exterior_compute_tags<
              GeneralizedHarmonic::Tags::ConstraintGamma0Compute<volume_dim,
                                                                 frame>,
              GeneralizedHarmonic::Tags::ConstraintGamma1Compute<volume_dim,
                                                                 frame>,
              GeneralizedHarmonic::Tags::ConstraintGamma2Compute<volume_dim,
                                                                 frame>,
              GeneralizedHarmonic::CharacteristicFieldsCompute<volume_dim,
                                                               frame>,
              GeneralizedHarmonic::CharacteristicSpeedsCompute<volume_dim,
                                                               frame>>,
          false>,
      Initialization::Actions::Evolution<EvolutionMetavars>,
      GeneralizedHarmonic::Actions::InitializeGaugeTags<volume_dim>,
      GeneralizedHarmonic::Actions::InitializeConstraintsTags<volume_dim>,
      dg::Actions::InitializeMortars<EvolutionMetavars, false>,
      Initialization::Actions::DiscontinuousGalerkin<EvolutionMetavars>,
      Initialization::Actions::Minmod<volume_dim>,
      Initialization::Actions::RemoveOptionsAndTerminatePhase>;

  using component_list = tmpl::flatten<tmpl::list<
      observers::Observer<EvolutionMetavars>,
      observers::ObserverWriter<EvolutionMetavars>,
      intrp::Interpolator<EvolutionMetavars>,
      intrp::InterpolationTarget<EvolutionMetavars, Horizon>,
      tmpl::conditional_t<is_numerical_initial_data_v<initial_data>,
                          importer::DataFileReader<EvolutionMetavars>,
                          tmpl::list<>>,
      DgElementArray<
          EvolutionMetavars,
          tmpl::list<
              Parallel::PhaseActions<Phase, Phase::Initialization,
                                     initialization_actions>,
              Parallel::PhaseActions<
                  Phase, Phase::InitializeTimeStepperHistory,
                  tmpl::flatten<tmpl::list<SelfStart::self_start_procedure<
                      compute_rhs, update_variables>>>>,
              Parallel::PhaseActions<
                  Phase, Phase::Register,
                  tmpl::flatten<tmpl::list<
                      intrp::Actions::RegisterElementWithInterpolator,
                      observers::Actions::RegisterWithObservers<
                          observers::RegisterObservers<
                              element_observation_type>>,
                      tmpl::conditional_t<
                          is_numerical_initial_data_v<initial_data>,
                          importer::Actions::RegisterWithImporter,
                          tmpl::list<>>,
                      Parallel::Actions::TerminatePhase>>>,
              Parallel::PhaseActions<
                  Phase, Phase::Evolve,
                  tmpl::flatten<tmpl::list<
                      Actions::RunEventsAndTriggers,
                      tmpl::conditional_t<
                          local_time_stepping,
                          Actions::ChangeStepSize<step_choosers>, tmpl::list<>>,
                      compute_rhs, update_variables,
                      Actions::AdvanceTime>>>>>>>;

  static constexpr OptionString help{
      "Evolve a generalized harmonic analytic solution.\n\n"
      "The analytic solution is: KerrSchild\n"
      "The numerical flux is:    UpwindFlux\n"};

  static Phase determine_next_phase(
      const Phase& current_phase,
      const Parallel::CProxy_ConstGlobalCache<
          EvolutionMetavars>& /*cache_proxy*/) noexcept {
    switch (current_phase) {
      case Phase::Initialization:
        return Phase::InitializeTimeStepperHistory;
      case Phase::InitializeTimeStepperHistory:
        return Phase::Register;
      case Phase::Register:
        return is_numerical_initial_data_v<initial_data> ? Phase::ImportData
                                                         : Phase::Evolve;
      case Phase::ImportData:
        return Phase::Evolve;
      case Phase::Evolve:
        return Phase::Exit;
      case Phase::Exit:
        ERROR(
            "Should never call determine_next_phase with the current phase "
            "being 'Exit'");
      default:
        ERROR(
            "Unknown type of phase. Did you static_cast<Phase> an integral "
            "value?");
    }
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling,
    &domain::creators::register_derived_with_charm,
    &Parallel::register_derived_classes_with_charm<MathFunction<1>>,
    &Parallel::register_derived_classes_with_charm<
        Event<metavariables::events>>,
    &Parallel::register_derived_classes_with_charm<
        StepChooser<metavariables::step_choosers>>,
    &Parallel::register_derived_classes_with_charm<StepController>,
    &Parallel::register_derived_classes_with_charm<TimeStepper>,
    &Parallel::register_derived_classes_with_charm<
        Trigger<metavariables::triggers>>};

static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
