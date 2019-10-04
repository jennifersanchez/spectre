// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <tuple>
#include <utility>  // IWYU pragma: keep
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Norms.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Evolution/Initialization/DiscontinuousGalerkin.hpp"
#include "Evolution/Initialization/Evolution.hpp"
#include "Evolution/Initialization/Limiter.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Characteristics.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticData/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeGhQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace GeneralizedHarmonic {
namespace Actions {
template <size_t Dim>
struct InitializeConstraintsTags {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        GeneralizedHarmonic::Tags::GaugeConstraintCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::FConstraintCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TwoIndexConstraintCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::FourIndexConstraintCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintEnergyCompute<Dim, frame>,

        // following tags added to observe constraints
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::GaugeConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::FConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::TwoIndexConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::ThreeIndexConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::FourIndexConstraint<Dim, frame>>,
        ::Tags::PointwiseL2NormCompute<
            GeneralizedHarmonic::Tags::ConstraintEnergy<Dim, frame>>>;

    return std::make_tuple(
        Initialization::merge_into_databox<InitializeConstraintsTags,
                                           db::AddSimpleTags<>, compute_tags>(
            std::move(box)));
  }
};

template <size_t Dim>
struct InitializeGHAnd3Plus1VariablesTags {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    using compute_tags = db::AddComputeTags<
        gr::Tags::SpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::DetAndInverseSpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::ShiftCompute<Dim, frame, DataVector>,
        gr::Tags::LapseCompute<Dim, frame, DataVector>,
        gr::Tags::SqrtDetSpatialMetricCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeNormalOneFormCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeNormalVectorCompute<Dim, frame, DataVector>,
        gr::Tags::InverseSpacetimeMetricCompute<Dim, frame, DataVector>,
        GeneralizedHarmonic::Tags::DerivSpatialMetricCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::DerivLapseCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::DerivShiftCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TimeDerivSpatialMetricCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TimeDerivLapseCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TimeDerivShiftCompute<Dim, frame>,
        gr::Tags::DerivativesOfSpacetimeMetricCompute<Dim, frame>,
        gr::Tags::DerivSpacetimeMetricCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::ThreeIndexConstraintCompute<Dim, frame>,
        gr::Tags::SpacetimeChristoffelFirstKindCompute<Dim, frame, DataVector>,
        gr::Tags::SpacetimeChristoffelSecondKindCompute<Dim, frame, DataVector>,
        gr::Tags::TraceSpacetimeChristoffelFirstKindCompute<Dim, frame,
                                                            DataVector>,
        gr::Tags::SpatialChristoffelFirstKindCompute<Dim, frame, DataVector>,
        gr::Tags::SpatialChristoffelSecondKindVarsCompute<Dim, frame,
                                                          DataVector>,
        ::Tags::DerivCompute<
            ::Tags::Variables<tmpl::list<gr::Tags::SpatialChristoffelSecondKind<
                Dim, frame, DataVector>>>,
            ::Tags::InverseJacobian<::Tags::ElementMap<Dim, frame>,
                                    ::Tags::LogicalCoordinates<Dim>>>,
        gr::Tags::TraceSpatialChristoffelFirstKindCompute<Dim, frame,
                                                          DataVector>,
        gr::Tags::RicciTensorCompute<Dim, frame, DataVector>,
        GeneralizedHarmonic::Tags::ExtrinsicCurvatureCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::TraceExtrinsicCurvatureCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintGamma0Compute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintGamma1Compute<Dim, frame>,
        GeneralizedHarmonic::Tags::ConstraintGamma2Compute<Dim, frame>>;

    template <typename TagsList>
    static auto initialize(
        db::DataBox<TagsList> && box,
        const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
        const double /*initial_time*/) noexcept {
      const size_t num_grid_points =
          db::get<::Tags::Mesh<Dim>>(box).number_of_grid_points();
      const auto& inertial_coords =
          db::get<::Tags::Coordinates<Dim, frame>>(box);

      // The evolution variables, gauge source function, and spacetime
      // derivative of the gauge source function will be overwritten by
      // numerical initial data. So here just initialize them to signaling nans.
      const auto& spacetime_metric =
          make_with_value<tnsr::aa<DataVector, Dim, frame>>(
              inertial_coords, std::numeric_limits<double>::signaling_NaN());
      const auto& phi = make_with_value<tnsr::iaa<DataVector, Dim, frame>>(
          inertial_coords, std::numeric_limits<double>::signaling_NaN());
      const auto& pi = make_with_value<tnsr::aa<DataVector, Dim, frame>>(
          inertial_coords, std::numeric_limits<double>::signaling_NaN());

      using Vars = db::item_type<variables_tag>;
      Vars vars{num_grid_points};
      const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<Dim>,
                                GeneralizedHarmonic::Tags::Phi<Dim>,
                                GeneralizedHarmonic::Tags::Pi<Dim>>
          solution_tuple(spacetime_metric, phi, pi);

      vars.assign_subset(solution_tuple);

      const auto& initial_gauge_source =
          make_with_value<tnsr::a<DataVector, Dim, frame>>(
              inertial_coords, std::numeric_limits<double>::signaling_NaN());
      const auto& spacetime_deriv_initial_gauge_source =
          make_with_value<tnsr::ab<DataVector, Dim, frame>>(
              inertial_coords, std::numeric_limits<double>::signaling_NaN());

      return db::create_from<db::RemoveTags<>, simple_tags, compute_tags>(
          std::move(box), std::move(vars), std::move(initial_gauge_source),
          std::move(spacetime_deriv_initial_gauge_source), 0.0, 1.0e30, 5.0e30);
    }
  };

template <size_t Dim>
struct InitializeGaugeTags {
  using frame = Frame::Inertial;

  template <typename DbTagsList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    // compute initial-gauge related quantities
    const auto mesh = db::get<::Tags::Mesh<Dim>>(box);
    const size_t num_grid_points = mesh.number_of_grid_points();
    const auto& lapse = get<gr::Tags::Lapse<DataVector>>(box);
    const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataVector>>>(box);
    const auto& deriv_lapse = get<
        ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<Dim>, frame>>(
        box);
    const auto& shift = get<gr::Tags::Shift<Dim, frame, DataVector>>(box);
    const auto& dt_shift =
        get<::Tags::dt<gr::Tags::Shift<Dim, frame, DataVector>>>(box);
    const auto& deriv_shift =
        get<::Tags::deriv<gr::Tags::Shift<Dim, frame, DataVector>,
                          tmpl::size_t<Dim>, frame>>(box);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<Dim, frame, DataVector>>(box);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(box);
    const auto& trace_christoffel_last_indices =
        get<gr::Tags::TraceSpatialChristoffelFirstKind<Dim, frame, DataVector>>(
            box);

    // call compute item for the initial gauge source function
    const auto initial_gauge_h = GeneralizedHarmonic::gauge_source<Dim, frame>(
        lapse, dt_lapse, deriv_lapse, shift, dt_shift, deriv_shift,
        spatial_metric, trace_extrinsic_curvature,
        trace_christoffel_last_indices);
    // set time derivatives of InitialGaugeH = 0
    // NOTE: this will need to be generalized to handle numerical initial data
    // and analytic initial data whose gauge is not initially stationary.
    const auto dt_initial_gauge_source =
        make_with_value<tnsr::a<DataVector, Dim, frame>>(lapse, 0.);

    // compute spatial derivatives of InitialGaugeH
    using InitialGaugeHVars = ::Variables<
        tmpl::list<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>>;
    InitialGaugeHVars initial_gauge_h_vars{num_grid_points};

    get<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>>(
        initial_gauge_h_vars) = initial_gauge_h;
    const auto inverse_jacobian = db::get<
        ::Tags::InverseJacobian<::Tags::ElementMap<Dim, frame>,
                                ::Tags::Coordinates<Dim, Frame::Logical>>>(box);
    const auto d_initial_gauge_source =
        get<::Tags::deriv<GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
                          tmpl::size_t<Dim>, frame>>(
            partial_derivatives<typename InitialGaugeHVars::tags_list>(
                initial_gauge_h_vars, mesh, inverse_jacobian));

    // compute spacetime derivatives of InitialGaugeH
    const auto initial_d4_gauge_h =
        GeneralizedHarmonic::Tags::SpacetimeDerivGaugeHCompute<
            Dim, frame>::function(dt_initial_gauge_source,
                                  d_initial_gauge_source);
    // Add all gauge tags
    using simple_tags = db::AddSimpleTags<
        GeneralizedHarmonic::Tags::InitialGaugeH<Dim, frame>,
        GeneralizedHarmonic::Tags::SpacetimeDerivInitialGaugeH<Dim, frame>>;
    using compute_tags = db::AddComputeTags<
        GeneralizedHarmonic::DampedHarmonicHCompute<Dim, frame>,
        GeneralizedHarmonic::SpacetimeDerivDampedHarmonicHCompute<Dim, frame>,
        GeneralizedHarmonic::Tags::DerivGaugeHFromSpacetimeDerivGaugeHCompute<
            Dim, frame>>;

    // Finally, insert gauge related quantities to the box
    return std::make_tuple(
        Initialization::merge_into_databox<InitializeGaugeTags, simple_tags,
                                           compute_tags>(
            std::move(box), std::move(initial_gauge_h),
            std::move(initial_d4_gauge_h)));
  }
};

}  // namespace Actions
}  // namespace GeneralizedHarmonic
