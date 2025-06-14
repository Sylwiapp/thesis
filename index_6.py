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

from idtxl.bivariate_te import BivariateTE    # IDTxl: multivariate transfer entropy class
from idtxl.visualise_graph import plot_network      # IDTxl: plotting class

# # Check if GPU is available
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("Using GPU for computation.")
# else:
# device = torch.device("cpu")
# print("GPU not available. Using CPU for computation.")

# Set the device for IDTxl
network_analysis = BivariateTE()
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
sys.stdout = open(f"logs/index6_alpha001.txt", "w")


# Rest of the code...
srcDir = '/home/syl/Documents/master/sylw_pociete/output_art'
subCode = 'ARZ000'
cond = 'art_watch2'
samplingRate = 1000 # ?    potestowac
samplesPerMs = samplingRate / 1000
eeg = loadRawEEG_epochs(srcDir, subCode, cond)

#zmiana 9.01 odkomentowuje blok
epochLengthMs = 1000  #1000
epochIndicesList = range(10,20)
data = adjustSignalToIDTxl(eeg, containesEpochedData=True, epochIndices=epochIndicesList)

#na ilu procesorach robie, zrownoleglic to XX 
# htop poleca narzedzie, top tez pokazuje 
# import threading, subproces 

#09.01
#data = adjustSignalToIDTxl(eeg, containesEpochedData=True)

# setup TE analysis
minLagInMs = 5
maxLagInMs = 30  # ile ms wstecz sprawdzac  #bylo 50

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
    "timeRange": [0, 11000],  #1/10
    "pastSpan": 1000,#50,  #0 lub all, ale wszystkie; number of samples to look into the past
    "step": 100,  #1,  50 - 100
    "targets": [7], #,],   czy dobrze robie i oba sa target i source?     #ustaw tu for ze jedna liczba jest targetem, a reszta sourcem i potem daleej w forze, ze kolejny targetem jest kolejnym i reszta sorcem
    #funkcja set do tego 
#ile czasu 
    "sources": [34],#,30,31,24,25,26,61,62],  # list of sources
    "cmi_estimator": "JidtGaussianCMI", #box-kernel - biased, kraskov- slower, best
    "fdr_correction": True,
}
#1. multivariate -> bivariate bez zmiany parametrow - ile sie liczy dla 20, ile dla 1sorce'a


#2. pastSpan": 100,
  #  "step": 50, 
#3. czy dane moge wrzucić - 1 czlwieka dane ew
#statystyczne moge 
#fdr correction :false moge 
#sroda 16.15 ->discord
#jednego badanego moge wrzucic, dobra tabelka

#chcemy miec liczbe zmiennoprzecinkową te 
#kod na githuba
# biblioteki threding, subprocess, na 10 rdzeniach 

#ustepstwa: bivariate miedzy 20? bez sensu
# uruchomic u michala 1target 1sensor/source
#

# Run moving TE analysis
resultList, aux = computeMovingBivariateTransferEntropy(data, moving_te_settings)
end_time = time.time()
execution_time = end_time - start_time
print(f"Czas wykonania programu(bez wyswietlenia matrixa): {execution_time} sekund")

#resultList.print_edge_list(weights='max_te_lag', fdr=False)
#plot_network(results=resultList, weights='max_te_lag', fdr=False)
#plt.show()


# # Create TE matrix for the last window
# num_targets = len(moving_te_settings["targets"])
# num_sources = len(moving_te_settings["sources"])
# te_matrix = np.zeros((num_sources, num_targets))
# last_result = resultList[-1]




for target in moving_te_settings["targets"]:
    plotSingleTargetMteTimeSeries(resultList, target)


# for target in moving_te_settings["targets"]:
    
#     mTE_vs_source_time = createSourceTargetMteDict(resultList, aux, target=target)

#     for source in moving_te_settings["sources"]:
#         plotSigleSourceTargetMteTimeSeries(
#             mTE_vs_source_time, source=source, targetLabel=str(target)
#         )


# # Plot inferred network to console and via matplotlib for the last window
# resultList.print_edge_list(weights='max_te_lag', fdr=False)
# plot_network(results=resultList, weights='max_te_lag', fdr=False)
# plt.show()
# input('Script ended. Press ENTER ...')

#zrob rest i oczy otwarte
#markery w plikach 

#dla jednego chłopka kod uruchamia sie na te parametry, podzielony 

# przetestowac transfer entrophy na dluzszym sygnale tj np 5sekundowym 
# jeszcze gpu sprobowac