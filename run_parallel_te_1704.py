from multiprocessing import Pool
import time
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network
from toolkit2 import setCwdHere, loadIDTxl, loadRawEEG
import torch

# Ustawienia globalne (przykładowe, dostosuj według potrzeb)
settings = {
    'cmi_estimator': 'JidtGaussianCMI',
    'n_perm_max_stat': 21,
    'n_perm_min_stat': 21,
    'n_perm_omnibus': 21,
    'n_perm_max_seq': 21,
    'max_lag_sources': 50,
    'min_lag_sources': 10,  # int(10 * (1000/1000))
    "alpha_min_stat": 0.5,
    "alpha_max_stat": 0.5,
    "alpha_omnibus": 0.5,
    "alpha_max_seq": 0.5,
    'pastSpan': 100,
    'step': 1000,
    'verbose': True
}

sources = [0, 1]
targets = [22, 23]

setCwdHere()
loadIDTxl()
network_analysis = MultivariateTE()

# Funkcja przetwarzająca pojedynczą epokę
def process_epoch(epoch):
    data_array = epoch.get_data() * 1e6  # skalowanie do µV
    data = Data(data_array, dim_order='ps', normalise=True, seed=1)
    result = network_analysis.analyse_network(settings=settings,
                                                data=data,
                                                sources=sources,
                                                targets=targets)
    # Dla diagnostyki wypisujemy wyniki epoki (opcjonalnie zapisz do logu)
    result.print_edge_list(weights='max_te_lag', fdr=False)
    return result

def main():
    # Wczytanie EEG i przygotowanie epok 
    srcDir = ''
    subCode = 'RGA798'
    cond = 'art_watch2'
    samplingRate = 1000
    fs = samplingRate
    eeg = loadRawEEG(srcDir, subCode, cond)
    events, event_id = mne.events_from_annotations(eeg)
    
    p_codes = [code for key, code in event_id.items() if key.startswith("Response/P")]
    m_codes = [code for key, code in event_id.items() if key.startswith("Response/M")]
    
    if not p_codes or not m_codes:
        raise ValueError("Markers 'Response/P' or 'Response/M' not found.")
        
    p_events = events[np.isin(events[:, 2], p_codes)][:, 0]
    m_events = events[np.isin(events[:, 2], m_codes)][:, 0]
    
    epoch_list = []
    for p_time in p_events:
        m_time = m_events[m_events > p_time]
        if len(m_time) == 0:
            continue
        m_time = m_time[0]
        epoch = eeg.copy().crop(tmin=p_time / fs, tmax=m_time / fs)
        epoch_list.append(epoch)
    
    print(f"Liczba epok: {len(epoch_list)}")
    
    # Mierzymy czas analizy
    start_time = time.time()
    
    # Tworzymy pulę procesów – używamy 12 procesów
    with Pool(processes=12) as pool:
        results_list = pool.map(process_epoch, epoch_list)
    
    total_time = time.time() - start_time
    print(f"Całkowity czas analizy: {total_time:.2f} sekund")
    
    # Tu można dodać zebranie wyników i wykonanie wizualizacji, np. histogram lagów
    combined_edges = []
    for res in results_list:
        # Wyciągamy macierz lagów
        try:
            adj_matrix_lag = res.get_adjacency_matrix(weights='max_te_lag', fdr=False)
            # Niech to będzie np. macierz NumPy - zakładamy, że jest kompatybilna
            adj_matrix_lag = np.asarray(adj_matrix_lag.todense())
            # Iterujemy po elementach macierzy
            for src in range(adj_matrix_lag.shape[0]):
                for tgt in range(adj_matrix_lag.shape[1]):
                    if adj_matrix_lag[src, tgt] != 0:
                        combined_edges.append((src, tgt, adj_matrix_lag[src, tgt]))
        except Exception as e:
            print(f"Błąd przy zbieraniu wyników: {e}")
    
    # Zapisujemy wyniki zbiorcze
    with open("logs/combined_results.txt", "w") as f:
        for edge in combined_edges:
            f.write(f"{edge}\n")
    
    # Przykładowa wizualizacja – histogram lagów
    if combined_edges:
        lags = [edge[2] for edge in combined_edges]
        plt.hist(lags, bins=20, alpha=0.7)
        plt.xlabel("Lag [ms]")
        plt.ylabel("Liczba połączeń")
        plt.title("Histogram lagów (połączone wyniki)")
        plt.grid(True)
        plt.savefig("plots/combined_lag_histogram.png")
        print("Wyniki zapisano do logs/combined_results.txt oraz plots/combined_lag_histogram.png")
    else:
        print("Brak znaczących połączeń.")

if __name__ == '__main__':
    main()
