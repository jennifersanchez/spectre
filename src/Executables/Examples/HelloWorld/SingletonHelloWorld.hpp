// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once
#include <cmath>

/// \cond
/// [executable_example_includes]
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"
/// [executable_example_includes]

/// [executable_example_options]
namespace OptionTags {
struct InitialGuess {
  using type = double;
  static constexpr OptionString help{"Initial Guess"};
};
struct RootSolve {
  using type = double;
  static constexpr OptionString help{"Root Solve"};
};
}  // namespace OptionTags

namespace Tags {
struct InitialGuess : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::InitialGuess>;

  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& initial_guess) noexcept {
    return initial_guess;
  }
};
namespace {
const double lower_bound = 0.0;
const double upper_bound = 10.0;
const size_t digits = 2;
const size_t max_iterations = 50;
const auto func_and_deriv = [](double x) noexcept {
  return std::make_pair(x, 1);
};
}  // namespace
struct RootSolve : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::RootSolve>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& root_solve) noexcept {
    return double(::RootFinder::newton_raphson(
        func_and_deriv, const double* initial_guess, lower_bound, upper_bound,
        digits));
  };
};
}  // namespace Tags
// namespace Tags
/// [executable_example_options]

/// [executable_example_action]
namespace Actions {
struct ComputeAndPrint {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    Parallel::printf("The answer is: %1.15f\n",
                     Parallel::get<Tags::InitialGuess>(cache),
                     Parallel::get<Tags::RootSolve>(cache));
  }
};
}  // namespace Actions
/// [executable_example_action]

/// [executable_example_singleton]
template <class Metavariables>
struct HelloWorld {
  using const_global_cache_tags = tmpl::list<Tags::InitialGuess>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Execute, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept;
};

template <class Metavariables>
void HelloWorld<Metavariables>::execute_next_phase(
    const typename Metavariables::Phase /* next_phase */,
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
  Parallel::simple_action<Actions::ComputeAndPrint>(
      Parallel::get_parallel_component<HelloWorld>(
          *(global_cache.ckLocalBranch())));
}
/// [executable_example_singleton]

/// [executable_example_metavariables]
struct Metavars {
  using component_list = tmpl::list<HelloWorld<Metavars>>;

  static constexpr OptionString help{
      "Say hello from a singleton parallel component."};

  enum class Phase { Initialization, Execute, Exit };

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        Metavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Execute
                                                  : Phase::Exit;
  }
};
/// [executable_example_metavariables]

/// [executable_example_charm_init]
static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// [executable_example_charm_init]
/// \endcond
