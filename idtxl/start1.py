import os, sys
import matplotlib.pyplot as plt
from mne import make_fixed_length_epochs
from toolkit import *

setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE    # IDTxl: mutlivariate transfer entropy class
from idtxl.visualise_graph import plot_network      # IDTxl: plotting class


# Single subject analysis

srcDir = ''
subCode = 'RGA798'
cond = 'art_watch1'
samplingRate = 1000 # Hz
samplesPerMs = samplingRate/1000


eeg = loadRawEEG(srcDir, subCode, cond)
epochLengthMs = 1000
# optionally narrow down epoch selection (for faster testing)
epochIndicesList = range(100,111)   

if epochLengthMs > 0:
    eeg = make_fixed_length_epochs(eeg, duration=epochLengthMs/1000, preload=True) # preload=False ?  
    data = adjustSignalToIDTxl(eeg, containesEpochedData=True, epochIndices=epochIndicesList)
else:
    data = adjustSignalToIDTxl(eeg, containesEpochedData=False)

# setup TE analysis
minLagInMs = 1
maxLagInMs = 2

network_analysis = MultivariateTE()
settings = {
    # list of parameters: https://pwollstadt.github.io/IDTxl/html/idtxl.html?highlight=analyse_single_target#idtxl.bivariate_mi.BivariateMI.analyse_single_target
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 50,
#    'n_perm_min_stat': 50,
#    'n_perm_omnibus': 50,
#    'n_perm_max_seq': 50,
    'max_lag_sources': int(maxLagInMs*samplesPerMs),
    'min_lag_sources': int(minLagInMs*samplesPerMs)
}

# Run analysis
results = network_analysis.analyse_network(
    settings=settings,
    data=data,
    sources=[1, 2,3,4,5,6], #[1, 2],  # 'all'
    targets='all'  #[4]  # 'all'
)

# Plot inferred network to console and via matplotlib
results.print_edge_list(weights='max_te_lag', fdr=False)
plot_network(results=results, weights='max_te_lag', fdr=False)
plt.show()
input('Script ended. Press ENTER ...')
