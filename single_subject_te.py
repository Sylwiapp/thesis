import os, sys
import matplotlib.pyplot as plt
from mne import make_fixed_length_epochs
from toolkit import *
import torch
import time

start_time = time.time()
setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE    # IDTxl: mutlivariate transfer entropy class
from idtxl.visualise_graph import plot_network      # IDTxl: plotting class

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU for computation.")
else:
    device = torch.device("cpu")
    print("GPU not available. Using CPU for computation.")

#Set the device for IDTxl
network_analysis = MultivariateTE()
network_analysis.set_device(device)

# Rest of the code...
srcDir = ''
subCode = 'RGA798'
cond = 'art_watch2'
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
maxLagInMs = 20 # ile ms wstecz >

settings = {
    # list of parameters: https://pwollstadt.github.io/IDTxl/html/idtxl.html?highlight=analyse_single_target#idtxl.bivariate_mi.BivariateMI.analyse_single_target
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 10,
   'n_perm_min_stat': 10,
    'n_perm_omnibus': 10,
   'n_perm_max_seq': 10,
 #   'max_lag_sources': 5,
    #'min_lag_sources': 1
     'max_lag_sources': int(maxLagInMs*samplesPerMs),
    'min_lag_sources': int(minLagInMs*samplesPerMs),
    "alpha_min_stat": 0.9,
    "alpha_max_stat": 0.9,
    "alpha_omnibus": 0.9,
    "alpha_max_seq": 0.9,
}

# # Create a list of all sources from 0 to 64
# sources = list(range(64))
# visual_electrodes = [29,30,31]  # actual indices for O1, O2, Oz

# # Remove visual_electrodes numbers from sources
# for electrode in visual_electrodes:
#     sources.remove(electrode)

# Run analysis
results = network_analysis.analyse_network(
    settings=settings,
    data=data,
    sources = 'all', #sources,
    targets= 'all' #visual_electrodes
)

# Initialize a 64x64 matrix with zeros
te_matrix = np.zeros((4,4))

# Populate the matrix with transfer entropy values
for target in range(4):
    try:
        single_target_results = results.get_single_target(target, fdr=False)
        sources = single_target_results.selected_vars_sources
        te_values = single_target_results.selected_sources_te
        for source, te in zip(sources, te_values):
            te_matrix[source[0], target] = te
    except RuntimeError as e:
        print(f"No results for target {target}: {e}")

# Print the matrix
print(" ")
print("Transfer Entropy Matrix (4x4):")
print(te_matrix)


end_time = time.time()
execution_time = end_time - start_time
print(f"Czas wykonania programu: {execution_time} sekund")

# Plot inferred network to console and via matplotlib
results.print_edge_list(weights='max_te_lag', fdr=False)
plot_network(results=results, weights='max_te_lag', fdr=False)
plt.show()
input('Script ended. Press ENTER ...')


