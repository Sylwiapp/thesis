import os, sys
import matplotlib.pyplot as plt
from mne import make_fixed_length_epochs
from toolkit import *
import torch
import time
import glob


start_time = time.time()
setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE    # IDTxl: multivariate transfer entropy class
from idtxl.visualise_graph import plot_network      # IDTxl: plotting class

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU for computation.")

# Set the device for IDTxl
network_analysis = MultivariateTE()
network_analysis.set_device(device)

# Find the highest numbered log file
log_files = glob.glob('logs_*.txt')
highest_num = 0
for file in log_files:
    num = int(file.split('_')[1].split('.')[0])
    if num > highest_num:
        highest_num = num

# Increment the number for the new log file
new_num = highest_num + 1

# Open the new log file for writing
sys.stdout = open(f"logs_{new_num}.txt", "w")


# Rest of the code...
srcDir = ''
subCode = 'ARZ000'
cond = 'art_watch2'
samplingRate = 1000 # Hz
samplesPerMs = samplingRate / 1000
eeg = loadRawEEG(srcDir, subCode, cond)

eeg.crop(tmin=11546 / samplingRate, tmax=22771 / samplingRate)

epochLengthMs = 0
# optionally narrow down epoch selection (for faster testing)
# epochIndicesList = range(100, 111)
if epochLengthMs > 0:
    eeg = make_fixed_length_epochs(eeg, duration=epochLengthMs / 1000, preload=True)  # preload=False ?  
    data = adjustSignalToIDTxl(eeg, containesEpochedData=True, epochIndices=epochIndicesList)
else:
    data = adjustSignalToIDTxl(eeg, containesEpochedData=False)

# setup TE analysis
minLagInMs = 1
maxLagInMs = 10  # ile ms wstecz sprawdzac

settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 100,
    'n_perm_min_stat': 100,
    'n_perm_omnibus': 100,
    'n_perm_max_seq': 100,
    'max_lag_sources': int(maxLagInMs * samplesPerMs),
    'min_lag_sources': int(minLagInMs * samplesPerMs),
    "alpha_min_stat": 0.05,
    "alpha_max_stat": 0.05,
    "alpha_omnibus": 0.05,
    "alpha_max_seq": 0.05,
}

# Moving multivariate TE analysis
moving_te_settings = {
    "timeRange": [0, data.data.shape[1] - 1],
    "pastSpan": 50,  #0 lub all, ale wszystkie; number of samples to look into the past
    "step": 1,  # number of samples to progress the calculation window
    "targets": [1], #,],   czy dobrze robie i oba sa target i source?     #ustaw tu for ze jedna liczba jest targetem, a reszta sourcem i potem daleej w forze, ze kolejny targetem jest kolejnym i reszta sorcem
    #funkcja set do tego 
#ile czasu 
    "sources": [2,3,4,5,6,7,12,13,14,15,16,23,24,25,26,27,29,31],  # list of sources
    "cmi_estimator": "JidtGaussianCMI", #box-kernel - biased, kraskov- slower, best
    "fdr_correction": True,
}

# Run moving TE analysis
resultList, aux = computeMovingMultivariateTransferEntropy(data, moving_te_settings)
end_time = time.time()
execution_time = end_time - start_time
print(f"Czas wykonania programu(bez wyswietlenia matrixa): {execution_time} sekund")

# Create TE matrix for the last window
num_targets = len(moving_te_settings["targets"])
num_sources = len(moving_te_settings["sources"])
te_matrix = np.zeros((num_sources, num_targets))
last_result = resultList[-1]


for target in moving_te_settings["targets"]:
    plotSingleTargetMteTimeSeries(resultList, target)


for target in moving_te_settings["targets"]:
    
    mTE_vs_source_time = createSourceTargetMteDict(resultList, aux, target=target)

    for source in moving_te_settings["sources"]:
        plotSigleSourceTargetMteTimeSeries(
            mTE_vs_source_time, source=source, targetLabel=str(target)
        )


# # Plot inferred network to console and via matplotlib for the last window
# resultList.print_edge_list(weights='max_te_lag', fdr=False)
# plot_network(results=resultList, weights='max_te_lag', fdr=False)
# plt.show()
# input('Script ended. Press ENTER ...')

#zrob rest i oczy otwarte
#markery w plikach 

#dla jednego chłopka kod uruchamia sie na te parametry, podzielony 

# przetestowac transfer entrophy na dluzszym sygnale tj np 5sekundowym 