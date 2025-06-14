#sprawdzanie bivariate, konfiguracja chatu
#->bivariate
#->ustawie 5 sourceów i targetow
#->dane poprawie
#param poprawie

import os, sys
import matplotlib.pyplot as plt
from mne import make_fixed_length_epochs
from toolkit2 import *
import torch
import time
import glob
from scipy.stats import zscore
from idtxl.data import Data

start_time = time.time()
setCwdHere()
loadIDTxl()

from idtxl.bivariate_te import BivariateTE    # IDTxl: multivariate transfer entropy class
from idtxl.visualise_graph import plot_network      # IDTxl: plotting class


network_analysis = BivariateTE()

sys.stdout = open(f"logs/index9_dziala2.txt", "w")


# Rest of the code...
srcDir = ''
subCode = 'RGA798'
cond = 'art_watch2'
samplingRate = 1000 # ?    potestowac
samplesPerMs = samplingRate / 1000

import mne
import numpy as np

# Wczytaj EEG
eeg = loadRawEEG(srcDir, subCode, cond)

# Pobierz eventy (markery)
events, event_id = mne.events_from_annotations(eeg)

# Sprawdź dostępne markery
print("Dostępne markery:", event_id, flush=True)

# 🔹 Filtrujemy tylko te eventy, które zawierają "Response/P" lub "Response/M"
p_codes = [code for key, code in event_id.items() if key.startswith("Response/P")]
m_codes = [code for key, code in event_id.items() if key.startswith("Response/M")]

# Sprawdzamy, czy znaleziono jakieś markery
if not p_codes or not m_codes:
    raise ValueError("Nie znaleziono żadnych markerów 'Response/P XXX' lub 'Response/M XXX' w pliku EEG.")

# Pobieramy próbki dla markerów P i M
p_events = events[np.isin(events[:, 2], p_codes)][:, 0]
m_events = events[np.isin(events[:, 2], m_codes)][:, 0]

print("Markery P (początek):", p_events, flush=True)
print("Markery M (koniec):", m_events, flush=True)

# Dopasowanie markerów P → M
epoch_list = []
fs = eeg.info["sfreq"]  # Częstotliwość próbkowania

for p_time in p_events:
    # Znajdź pierwsze `M`, które pojawia się po `P`
    m_time = m_events[m_events > p_time]
    if len(m_time) == 0:
        continue  # Jeśli nie ma końcowego `M`, pomijamy

    m_time = m_time[0]  # Najbliższy marker `M`

    # Wycinamy EEG od `P` do `M`
    epoch = eeg.copy().crop(tmin=p_time / fs, tmax=m_time / fs)
    epoch_list.append(epoch)

print(f"Znaleziono {len(epoch_list)} epok EEG.", flush=True)
#####################################################

# Wybierz pierwszą epokę
first_epoch = epoch_list[0].get_data()

# Wybierz elektrodę do sprawdzenia (np. Fz)
channel_idx = 0  # Możesz zmienić na inną elektrodę

# Oś czasu
fs = eeg.info["sfreq"]  # Częstotliwość próbkowania
time = np.arange(first_epoch.shape[1]) / fs  # Poprawione: shape[1] zamiast shape[2]

# Rysowanie sygnału EEG z pierwszej epoki
plt.figure(figsize=(10, 4))
plt.plot(time, first_epoch[channel_idx, :], label=f"Kanał {channel_idx}")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda (µV)")
plt.title("Pierwsza epoka EEG")
plt.legend()
#plt.show()

print("Minimalna wartość:", np.min(first_epoch), flush=True)
print("Maksymalna wartość:", np.max(first_epoch), flush=True)

#####################



# 📌 5️⃣ Analiza każdej epoki osobno za pomocą Bivariate TE
# Ustawienia dla IDTxl
settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 50,
    'n_perm_min_stat': 50,
    'n_perm_omnibus': 50,
    'n_perm_max_seq': 50,
    'max_lag_sources': int(100 * samplesPerMs),
    'min_lag_sources': int(50 * samplesPerMs),
    "alpha_min_stat": 0.2,
    "alpha_max_stat": 0.2,
    "alpha_omnibus": 0.2,
    "alpha_max_seq": 0.2,
    "pastSpan": 50,#50,  #0 lub all, ale wszystkie; number of samples to look into the past
    "step": 1000, 
    "verbose": True
}

# Lista wyników TE dla każdej epoki
results_list = []

for i, epoch in enumerate(epoch_list):
    print(f" Analizuję epokę {i+1}/{len(epoch_list)}", flush=True)

    data_array = epoch.get_data() * 1e6  # Przeskalowanie do µV
    #data_array = data_array.T 

    # Konwersja do IDTxl
    print(f"Przed konwersją do IDTxl, kształt danych: {data_array.shape}", flush=True)
    data = Data(data_array, dim_order='ps', normalise=True, seed=1)
    print(f"Konwersja do IDTxl zakończona dla epoki {i+1}", flush=True)

    # Uruchomienie analizy Bivariate TE
    try:
        print(f"Start analizy TE dla epoki {i+1}", flush=True)
        results = network_analysis.analyse_network(
            settings=settings,
            data=data,
            sources=[3],#4, 5, 42, 43],
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