"""Perform network inference using multivarate transfer entropy.

Estimate multivariate transfer entropy (TE) for network inference using a
greedy approach with maximum statistics to generate a non-uniform embedding
(Faes, 2011; Lizier, 2012).

Note:
    Written for Python 3.4+
"""
from .network_inference import NetworkInferenceTE, NetworkInferenceMultivariate
from .stats import network_fdr
from .results import ResultsNetworkInference


import torch

class MultivariateTE(NetworkInferenceTE, NetworkInferenceMultivariate):
    """Perform network inference using multivariate transfer entropy."""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cpu")  # Default to CPU
    
    def set_device(self, device):
        """Set the device for computation."""
        self.device = device
        print("Using GPU for computation.")
    
    def analyse_network(self, settings, data, targets="all", sources="all"):
        """Find multivariate transfer entropy between all nodes in the network."""
        # Set defaults for network inference.
        settings.setdefault("verbose", True)
        settings.setdefault("fdr_correction", True)

        # Check which targets and sources are requested for analysis.
        if targets == "all":
            targets = [t for t in range(data.n_processes)]
        if sources == "all":
            sources = ["all" for t in targets]
        if (type(sources) is list) and (type(sources[0]) is int):
            sources = [sources for t in targets]
        if (type(sources) is list) and (type(sources[0]) is list):
            pass
        else:
            ValueError("Sources was not specified correctly: {0}.".format(sources))
        assert len(sources) == len(targets), (
            "List of targets and list of "
            "sources have to have the same "
            "same length"
        )

        # Check and set defaults for checkpointing.
        settings = self._set_checkpointing_defaults(settings, data, sources, targets)

        # Perform TE estimation for each target individually
        results = ResultsNetworkInference(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(),
            normalised=data.normalise,
        )
        for t in range(len(targets)):
            if settings["verbose"]:
                print(
                    "\n####### analysing target with index {0} from list {1}".format(
                        t, targets
                    )
                )
            res_single = self.analyse_single_target(
                settings, data, targets[t], sources[t]
            )
            results.combine_results(res_single)

        # Get no. realisations actually used for estimation from single target
        # analysis.
        results.data_properties.n_realisations = (
            res_single.data_properties.n_realisations
        )

        # Perform FDR-correction on the network level. Add FDR-corrected
        # results as an extra field. Network_fdr/combine_results internally
        # creates a deep copy of the results.
        if settings["fdr_correction"]:
            results = network_fdr(settings, results)
        return results

    def analyse_single_target(self, settings, data, target, sources="all"):
        """Find multivariate transfer entropy between sources and a target."""
        # Check input and clean up object if it was used before.
        self._initialise(settings, data, sources, target)

        # Main algorithm.
        print("\n---------------------------- (1) include target candidates")
        self._include_target_candidates(data)
        print("\n---------------------------- (2) include source candidates")
        self._include_source_candidates(data)
        print("\n---------------------------- (3) prune source candidate")
        self._prune_candidates(data)
        print("\n---------------------------- (4) final statistics")
        self._test_final_conditional(data)

        # Clean up and return results.
        if self.settings["verbose"]:
            print(
                "final source samples: {0}".format(
                    self._idx_to_lag(self.selected_vars_sources)
                )
            )
            print(
                "final target samples: {0}\n\n".format(
                    self._idx_to_lag(self.selected_vars_target)
                )
            )
        results = ResultsNetworkInference(
            n_nodes=data.n_processes,
            n_realisations=data.n_realisations(self.current_value),
            normalised=data.normalise,
        )
        results._add_single_result(
            target=self.target,
            settings=self.settings,
            results={
                "sources_tested": self.source_set,
                "current_value": self.current_value,
                "selected_vars_target": self._idx_to_lag(self.selected_vars_target),
                "selected_vars_sources": self._idx_to_lag(self.selected_vars_sources),
                "selected_sources_pval": self.pvalues_sign_sources,
                "selected_sources_te": self.statistic_sign_sources,
                "omnibus_te": self.statistic_omnibus,
                "omnibus_pval": self.pvalue_omnibus,
                "omnibus_sign": self.sign_omnibus,
                "te": self.statistic_single_link,
            },
        )
        self._reset()  # remove attributes
        return results

    def __eq__(self, other):
        return self.target == other.target

    def getit(self):
        print(self.target)
