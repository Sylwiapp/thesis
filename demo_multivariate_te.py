# Import classes
from toolkit import *

setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
import matplotlib.pyplot as plt

# a) Generate test data
data = Data()
# data.generate_mute_data(n_samples=1000, n_replications=5)
data.set_data(testDataset("step2D"), dim_order="ps")
# data.set_data(testDataset("step3D"), dim_order="psr")
# data.set_data(testDataset("logistic-coupling"), dim_order="ps")

visualizeInputData(data=data)
# visualize2Processes(data=data)
pass

# b) Initialise analysis object and define settings
network_analysis = MultivariateTE()
# network_analysis = BivariateTE()
settings = {
    "cmi_estimator": "JidtGaussianCMI",
    "max_lag_sources": 3,
    "min_lag_sources": 1,
    # "fdr_correction": False,
    # "add_conditionals": [
    # #     # (0, 1),
    # #     # (0, 2),
    # #     # (0, 3),
    # #     # (0, 4),
    # #     # (0, 5),
    # #     # (0, 6),
    # #     # (0, 7),
    # #     # (1, 1),
    # #     # (1, 2),
    # #     # (1, 3),
    # #     # (1, 4),
    # #     # (1, 5),
    # #     # (1, 6),
    # #     # (1, 7),
    # # ],  # try to add all variables as relevant sources
}

# c) Run analysis
targetList = [1]  # [0, 1, 2, 3, 4]
results = network_analysis.analyse_network(
    settings=settings, data=data, targets=targetList
)

# d) show results
showResults(results, targets=targetList, withFDR=False)
