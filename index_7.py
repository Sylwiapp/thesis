#sprawdzanie multivariate wszystkich, za DLUGO
#
import os, sys
import matplotlib.pyplot as plt
from mne import make_fixed_length_epochs
from toolkit2 import *
import torch
import time
import glob


start_time = time.time()
setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE    # IDTxl: multivariate transfer entropy class
from idtxl.visualise_graph import plot_network      # IDTxl: plotting class

# # Check if GPU is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU for computation.")
# else:
# device = torch.device("cpu")
# print("GPU not available. Using CPU for computation.")

# Set the device for IDTxl
network_analysis = MultivariateTE()

# network_analysis.set_device(device)

# Find the highest numbered log file
# log_files = glob.glob('logs_*.txt')
# highest_num = 0
# for file in log_files:
#     num = int(file.split('_')[1].split('.')[0])
#     if num > highest_num:
#         highest_num = num

# Increment the number for the new log file
# new_num = highest_num + 1

# Open the new log file for writing
sys.stdout = open(f"logs/test_index7_005_all_18.txt", "w")


# Rest of the code...
srcDir = ''
subCode = 'RGA798'
cond = 'art_watch2'
samplingRate = 1000 # ?    potestowac
samplesPerMs = samplingRate / 1000

eeg = loadRawEEG(srcDir, subCode, cond)

#eeg.crop(tmin=16870 / samplingRate, tmax=24008 / samplingRate) 
eeg.crop(tmin=28078 / samplingRate, tmax=35200 / samplingRate) 

data = adjustSignalToIDTxl(eeg, containesEpochedData=False)

# setup TE analysis
minLagInMs = 5
maxLagInMs = 1000  # ile ms wstecz sprawdzac  #bylo 50

settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 50,
    'n_perm_min_stat': 50,
    'n_perm_omnibus': 50,
    'n_perm_max_seq': 50,
    'max_lag_sources': int(maxLagInMs * samplesPerMs),
    'min_lag_sources': int(minLagInMs * samplesPerMs),
    "alpha_min_stat": 0.2,
    "alpha_max_stat": 0.2,
    "alpha_omnibus": 0.2,
    "alpha_max_seq": 0.2,
    "pastSpan": 1000,#50,  #0 lub all, ale wszystkie; number of samples to look into the past
    "step": 1000, 
    "verbose": True
}

# Run analysis
results = network_analysis.analyse_network(
    settings=settings,
    data=data,
    sources=[14],#,15,16,23],
    targets=[24]#,25,26,27,29,31]
)

# Plot inferred network to console and via matplotlib
results.print_edge_list(weights='max_te_lag', fdr=False)
plot_network(results=results, weights='max_te_lag', fdr=False)
plt.show()
input('Script ended. Press ENTER ...')

#resultList.print_edge_list(weights='max_te_lag', fdr=False)
#plot_network(results=resultList, weights='max_te_lag', fdr=False)
#plt.show()


# # Create TE matrix for the last window
# num_targets = len(moving_te_settings["targets"])
# num_sources = len(moving_te_settings["sources"])
# te_matrix = np.zeros((num_sources, num_targets))
# last_result = resultList[-1]



