// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once
#include <cmath>
#include <math.h>

/// \cond
/// [executable_example_includes]
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Options/Options.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Utilities/TMPL.hpp"
/// [executable_example_includes]
namespace {
const auto func_and_deriv = [](double x) noexcept {
  return std::make_pair(sin(x), cos(x));
};
}  // namespace

/// [executable_example_options]
namespace OptionTags {
struct InitialGuess {
  using type = double;
  static constexpr OptionString help{"Initial Guess"};
};

struct LowerBound {
  using type = double;
  static constexpr OptionString help{"Lower Bound"};
};

struct UpperBound {
  using type = double;
  static constexpr OptionString help{"Upper Bound"};
};

struct Digits {
  using type = double;
  static constexpr OptionString help{"Digits"};
};

struct MaxIterations {
  using type = double;
  static constexpr OptionString help{"Max Iteration"};
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
struct LowerBound : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::LowerBound>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& lower_bound) noexcept {
    return lower_bound;
  }
};
struct UpperBound : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::UpperBound>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& upper_bound) noexcept {
    return upper_bound;
  }
};

struct Digits : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::Digits>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& digits) noexcept {
    return digits;
  }
};

struct MaxIterations : db::SimpleTag {
  using type = double;
  using option_tags = tmpl::list<OptionTags::MaxIterations>;
  static constexpr bool pass_metavariables = false;
  static double create_from_options(const double& max_iterations) noexcept {
    return max_iterations;
  }
};

struct RootSolve : db::SimpleTag {
  using type = double;
};
struct RootSolveCompute : RootSolve, db::ComputeTag {
  static double function(const double& initial_guess, const double& lower_bound,
                         const double& upper_bound, const double& digits,
                         const double& max_iterations) noexcept {
    return double(::RootFinder::newton_raphson(func_and_deriv, initial_guess,
                                               lower_bound, upper_bound, digits,
                                               max_iterations));
  }
  using argument_tags =
      tmpl::list<InitialGuess, LowerBound, UpperBound, Digits, MaxIterations>;
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
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/) {
    double result{::RootFinder::newton_raphson(
        func_and_deriv, Parallel::get<Tags::InitialGuess>(cache),
        Parallel::get<Tags::LowerBound>(cache),
        Parallel::get<Tags::UpperBound>(cache),
        Parallel::get<Tags::Digits>(cache),
        Parallel::get<Tags::MaxIterations>(cache))};
    Parallel::printf(
        "The initial guess: %1.15f, the lower bound: %1.15f, the upper bound: "
        "%1.15f, the digits are %1d, the maximum iterations are %2d, the "
        "answer is: "
        "%1.15f \n",
        Parallel::get<Tags::InitialGuess>(cache),
        Parallel::get<Tags::LowerBound>(cache),
        Parallel::get<Tags::UpperBound>(cache),
        Parallel::get<Tags::Digits>(cache),
        Parallel::get<Tags::MaxIterations>(cache), result);
  }
};
}  // namespace Actions
/// [executable_example_action]

/// [executable_example_singleton]
template <class Metavariables>
struct HelloWorld {
  using const_global_cache_tags =
      tmpl::list<Tags::InitialGuess, Tags::LowerBound, Tags::UpperBound,
                 Tags::Digits, Tags::MaxIterations>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Execute, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) noexcept;
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

  static constexpr Options::String help{
      "Say hello from a singleton parallel component."};

  enum class Phase { Initialization, Execute, Exit };

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_GlobalCache<
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
