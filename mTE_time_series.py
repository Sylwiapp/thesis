# this script attempts to do running-n transfer entropy estimation - to test whether i can see DYNAMIC infrmation transfer (via mTE) as time-series
# Import classes
from toolkit import *

setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE
from idtxl.bivariate_te import BivariateTE
from idtxl.data import Data
import matplotlib.pyplot as plt
import random
import copy

mySeed = 14
random.seed(mySeed)
np.random.seed(mySeed)

# a) Generate test data
data = Data(
    normalise=False, seed=mySeed
)  # !!! seed is used inside this constructor as numpy.random.seed(seed)
data.generate_mute_data(n_samples=32, n_replications=5)
# data.set_data(testDataset("linear"), dim_order="ps")
# data.set_data(testDataset("step2D"), dim_order="ps")
# data.set_data(testDataset("step3D"), dim_order="psr")
# data.set_data(testDataset("logistic-coupling"), dim_order="ps")

# TEST RNG
# print(getSignalHashString(data.data))

visualizeInputData(data=data, scatter=0)
# visualize2Processes(data=data)

pass

targetList = [1]
sourcesList = [0, 2]
nSamples = data.data.shape[1]

# c) run analysis sample-by-sample

settings = {
    "timeRange": [
        0,
        nSamples - 1,
    ],  # starting and ending sample index, or 'all' i.e. take whole time range of the signal
    "pastSpan": 10,  # number of samples to look into the past or 'all' which always takes maximal number of samples in the past available
    "step": 1,  # number of samples to progress the calculation window
    "targets": targetList,  # list of targets
    "sources": sourcesList,  # list of sources
    "cmi_estimator": "JidtGaussianCMI",  # see idtxl.network_analysis.analyse_network
    "fdr_correction": True,  # see idtxl.network_analysis.analyse_network
}

resultList, aux = computeMovingMultivariateTransferEntropy(data, settings) #data ma 3 wymiary, chennel, adata ,realizacje poradzi sovie z 3 i nic z nia nie zrobi

# plot total entropy time-series for single target
for target in targetList:
    plotSingleTargetMteTimeSeries(target=1, resultList=resultList)

# %% show as many plots as there are significant sources

mTE_vs_source_time = createSourceTargetMteDict(resultList, aux, target=1)

for source in sourcesList:
    plotSigleSourceTargetMteTimeSeries(
        mTE_vs_source_time, source=source, targetLabel="1"
    )


#4czerwca 16.45



