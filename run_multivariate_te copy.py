import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from scipy.stats import zscore
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE 
from idtxl.visualise_graph import plot_network

from toolkit2 import setCwdHere, loadIDTxl, loadRawEEG, plotSingleTargetMteTimeSeries  

setCwdHere()
loadIDTxl()

network_analysis = MultivariateTE()

sys.stdout = open(f"logs/logs_run_multivariate_te_1.txt", "w")

# Eeg data folder
srcDir = ''
subCode = 'RGA798'
cond = 'art_watch2'

samplingRate = 1000 
samplesPerMs = samplingRate / 1000

# Wczytaj EEG
eeg = loadRawEEG(srcDir, subCode, cond)

# Pobierz eventy (markery)
events, event_id = mne.events_from_annotations(eeg)

# Filtrujemy tylko te eventy, które zawierają "Response/P" lub "Response/M" 
# Marker P- rozpoczęcie oglądania obrazu, M-zakończenie
p_codes = [code for key, code in event_id.items() if key.startswith("Response/P")] 
m_codes = [code for key, code in event_id.items() if key.startswith("Response/M")]

if not p_codes or not m_codes:
    raise ValueError("Nie znaleziono żadnych markerów 'Response/P XXX' lub 'Response/M XXX' w pliku EEG.")

# Pobieramy próbki dla markerów P i M
p_events = events[np.isin(events[:, 2], p_codes)][:, 0]
m_events = events[np.isin(events[:, 2], m_codes)][:, 0]

print("Markery P (początek):", p_events, flush=True)
print("Markery M (koniec):", m_events, flush=True)

epoch_list = []
fs = eeg.info["sfreq"]

# Wycinamy EEG od `P` do `M`
for p_time in p_events:
    
    m_time = m_events[m_events > p_time]
    if len(m_time) == 0:
        continue  

    m_time = m_time[0]  
    # Dodanie to listy epok osobnych obrazów
    epoch = eeg.copy().crop(tmin=p_time / fs, tmax=m_time / fs)
    epoch_list.append(epoch)

print(f"Znaleziono {len(epoch_list)} epok EEG.", flush=True)
#####################################################
#Podejrzenie wyglądu sygnału

# first_epoch = epoch_list[0].get_data()
# channel_idx = 0 
# fs = eeg.info["sfreq"]  
# time = np.arange(first_epoch.shape[1]) / fs 

# # Rysowanie sygnału EEG z pierwszej epoki
# plt.figure(figsize=(10, 4))
# plt.plot(time, first_epoch[channel_idx, :], label=f"Kanał {channel_idx}")
# plt.xlabel("Czas (s)")
# plt.ylabel("Amplituda (µV)")
# plt.title("Pierwsza epoka EEG")
# plt.legend()
# plt.show()

#####################
#Aktualne ustawienia służą temu, żeby coś wyświetlać i żeby dość szybko się liczyło

settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 50,
    'n_perm_min_stat': 50,
    'n_perm_omnibus': 50,
    'n_perm_max_seq': 50,
    'max_lag_sources': int(1000 * samplesPerMs),
    'min_lag_sources': int(20 * samplesPerMs),
    "alpha_min_stat": 0.05,
    "alpha_max_stat": 0.05,
    "alpha_omnibus": 0.05,
    "alpha_max_seq": 0.05,
    "pastSpan": 1000,
    "step": 1000, 
    "verbose": True
}

# Lista wyników TE dla każdej epoki, gdzie epoką jest ok 7sekundowe oglądanie jednego obrazu
results_list = []

for i, epoch in enumerate(epoch_list):
    print(f" Analizuję epokę {i+1}/{len(epoch_list)}", flush=True)

    data_array = epoch.get_data() * 1e6  # Przeskalowanie do µV
    
    #NORMALIZACJA DANYCH
    data = Data(data_array, dim_order='ps', normalise=True, seed=1)

    # Uruchomienie analizy Bivariate TE
    try:
        print(f"Start analizy TE dla epoki {i+1}", flush=True)
        results = network_analysis.analyse_network(
            settings=settings,
            data=data,
            sources=[3, 4, 5, 42, 43],
            targets=[28, 30, 23, 25, 24]
        )
        print(f" Analiza TE zakończona dla epoki {i+1}", flush=True)
        results_list.append(results)
    except Exception as e:
        print(f"Błąd podczas analizy TE w epoce {i+1}: {e}", flush=True)
    
    print(f"\n Epoka {i+1}", flush=True)
    results.print_edge_list(weights='max_te_lag', fdr=False)

#  Wyświetlenie wyników dla każdej epoki
for i, results in enumerate(results_list):
    print(f"\n Epoka {i+1}", flush=True)
    results.print_edge_list(weights='max_te_lag', fdr=False)

#  Uśrednione TE dla wszystkich epok
all_te_matrices = np.array([r.get_adjacency_matrix(weights='max_te_lag') for r in results_list])
mean_te = np.mean(all_te_matrices, axis=0)


print("\n Średnia macierz TE dla wszystkich epok:")
print(mean_te)

# Wyświetlenie grafów
for target in settings["targets"]:
    plotSingleTargetMteTimeSeries(results_list, target)

# Plot inferred network to console and via matplotlib for the last window
results_list.print_edge_list(weights='max_te_lag', fdr=False)
plot_network(results=results_list, weights='max_te_lag', fdr=False)
plt.show()
input('Script ended. Press ENTER ...')