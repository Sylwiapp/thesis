import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mne import make_fixed_length_epochs
from toolkit import *

setCwdHere()
loadIDTxl()

from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network

# Single subject analysis
srcDir = ''
subCode = 'RGA798'
cond = 'art_watch1'
samplingRate = 1000  # Hz
samplesPerMs = samplingRate / 1000

eeg = loadRawEEG(srcDir, subCode, cond)
epochLengthMs = 1000
epochIndicesList = range(100, 111)  # or another range depending on your dataset size and needs

if epochLengthMs > 0:
    eeg = make_fixed_length_epochs(eeg, duration=epochLengthMs / 1000, preload=True)
    data = adjustSignalToIDTxl(eeg, containsEpochedData=True, epochIndices=epochIndicesList)
else:
    data = adjustSignalToIDTxl(eeg, containsEpochedData=False)

# setup TE analysis
minLagInMs = 1
maxLagInMs = 2

network_analysis = MultivariateTE()
settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 50,
    'max_lag_sources': int(maxLagInMs * samplesPerMs),
    'min_lag_sources': int(minLagInMs * samplesPerMs)
}

# Define number of channels and initialize TE matrix
num_channels = 64  # Number of EEG channels
num_epochs = len(epochIndicesList)
te_matrix = np.zeros((num_channels, num_channels, num_epochs))

# Run analysis
results = network_analysis.analyse_network(settings=settings, data=data, sources='all', targets='all')

# Populate the TE matrix with results
for t in range(num_channels):
    for s in range(num_channels):
        if t != s:  # Assuming TE from a channel to itself is not meaningful
            for k, epoch in enumerate(epochIndicesList):
                te_value = results.get_te_value(source=s, target=t, time=epoch)  # Hypothetical method
                te_matrix[s, t, k] = te_value

# Optionally, visualize and save the TE matrix
# For example, plot TE values for a specific pair over time
source_channel = 5
target_channel = 20
plt.plot(te_matrix[source_channel, target_channel, :])
plt.title(f'Transfer Entropy from Channel {source_channel} to Channel {target_channel}')
plt.xlabel('Epoch')
plt.ylabel('Transfer Entropy')
plt.show()

# Save the TE matrix to a file (optional)
np.save('te_matrix.npy', te_matrix)

input('Script ended. Press ENTER to exit...')
