// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
#include <cmath>
#include <cstddef>
#include <random>
#include <string>

#include "AlgorithmArray.hpp"
#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "ErrorHandling/FloatingPointExceptions.hpp"
#include "Options/Options.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/InitializationFunctions.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Parallel/Printf.hpp"
#include "Parallel/Reduction.hpp"
#include "Utilities/TMPL.hpp"

namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
template <class Metavariables>
struct PiMonteCarlo;
template <class Metavariables>
struct PiMonteCarloArray;

namespace {
size_t throw_darts(const size_t& number_of_darts_to_throw) noexcept {
  std::random_device device;         // actual random number for seed
  std::mt19937 generator(device());  // pseudorandom generator
  std::uniform_real_distribution<> distribution{0.0, 1.0};

  double x = 0;
  double y = 0;
  size_t hits = 0;
  for (size_t throws = 0; throws < number_of_darts_to_throw; ++throws) {
    x = distribution(generator);
    y = distribution(generator);
    if (square(x) + square(y) < 1.0) {
      ++hits;
    }
  }
  return hits;
}
}  // namespace

namespace OptionTags {
struct NumberOfThrows {
  using type = size_t;
  static constexpr OptionString help{"Number of darts to throw"};
};
}  // namespace OptionTags

namespace Tags {
struct NumberOfThrows : db::SimpleTag {
  using type = size_t;
  static std::string name() noexcept { return "NumberOfThrows"; };
  using option_tags = tmpl::list<OptionTags::NumberOfThrows>;

  template <typename Metavariables>
  static size_t create_from_options(const size_t& number_of_elements) noexcept {
    return number_of_elements;
  }
};
}  // namespace Tags

namespace Actions {
struct ProcessReducedSumOfHits {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(db::DataBox<DbTags>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const size_t& value) noexcept {
    // The total number of darts thrown might be less than Tags::NumberOfThrows
    // if the NumberOfThrows doesn't divide evenly into the number of
    // processors. The next line computes the number of darts actually thrown.
    const size_t total_darts_thrown =
        (Parallel::get<Tags::NumberOfThrows>(cache) /
         static_cast<size_t>(abs(Parallel::number_of_procs()))) *
        static_cast<size_t>(abs(Parallel::number_of_procs()));

    const double pi_estimate = 4.0 * static_cast<double>(value) /
                               static_cast<double>(total_darts_thrown);
    Parallel::printf("Pi estimate: %1.15f\n", pi_estimate);
    Parallel::printf("Abs(pi - estimate) = %1.15f\n", fabs(M_PI - pi_estimate));
    Parallel::printf("Darts thrown: %d", total_darts_thrown);
  }
};

struct ThrowDarts {
  template <typename ParallelComponent, typename DbTags, typename Metavariables,
            typename ArrayIndex>
  static void apply(const db::DataBox<DbTags>& /*box*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& array_index) noexcept {
    const auto& my_proxy =
        Parallel::get_parallel_component<PiMonteCarloArray<Metavariables>>(
            cache)[array_index];
    const auto& singleton_proxy =
        Parallel::get_parallel_component<PiMonteCarlo<Metavariables>>(cache);

    const size_t throws_per_element =
        Parallel::get<Tags::NumberOfThrows>(cache) /
        static_cast<size_t>(abs(Parallel::number_of_procs()));

    Parallel::ReductionData<Parallel::ReductionDatum<size_t, funcl::Plus<>>>
        hits{throw_darts(throws_per_element)};
    Parallel::contribute_to_reduction<ProcessReducedSumOfHits>(hits, my_proxy,
                                                               singleton_proxy);
  }
};
}  // namespace Actions

template <class Metavariables>
struct PiMonteCarlo {
  using const_global_cache_tags = tmpl::list<>;
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Execute, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  static void execute_next_phase(
      const typename Metavariables::Phase /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<Metavariables>&
      /*global_cache*/) noexcept {};
};

template <class Metavariables>
struct PiMonteCarloArray {
  using const_global_cache_tags = tmpl::list<Tags::NumberOfThrows>;
  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using array_index = int;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<typename Metavariables::Phase,
                             Metavariables::Phase::Execute, tmpl::list<>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void allocate_array(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const tuples::tagged_tuple_from_typelist<initialization_tags>&
      /*initialization_items*/) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& array_proxy =
        Parallel::get_parallel_component<PiMonteCarloArray<Metavariables>>(
            local_cache);

    for (size_t i = 0, which_proc = 0,
                number_of_procs =
                    static_cast<size_t>(abs(Parallel::number_of_procs()));
         i < number_of_procs; ++i) {
      array_proxy[i].insert(global_cache, {}, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
    array_proxy.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::Execute) {
      Parallel::simple_action<Actions::ThrowDarts>(
          Parallel::get_parallel_component<PiMonteCarloArray>(local_cache));
    }
  }
};

struct Metavars {
  using component_list =
      tmpl::list<PiMonteCarlo<Metavars>, PiMonteCarloArray<Metavars>>;

  static constexpr OptionString help{
      "Compute pi using Monte Carlo integration."};

  enum class Phase { Initialization, Execute, Exit };

  static Phase determine_next_phase(const Phase& current_phase,
                                    const Parallel::CProxy_ConstGlobalCache<
                                        Metavars>& /*cache_proxy*/) noexcept {
    return current_phase == Phase::Initialization ? Phase::Execute
                                                  : Phase::Exit;
  }
};

static const std::vector<void (*)()> charm_init_node_funcs{
    &setup_error_handling};
static const std::vector<void (*)()> charm_init_proc_funcs{
    &enable_floating_point_exceptions};
/// \endcond
