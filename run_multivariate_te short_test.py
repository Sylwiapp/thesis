import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE 
from idtxl.visualise_graph import plot_network
from toolkit2 import setCwdHere, loadIDTxl, loadRawEEG, plotSingleTargetMteTimeSeries  
import torch

def main():
    try:
        # Ustawienie urządzenia (GPU lub CPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'Using device: {device}', flush=True)
        
        setCwdHere()
        loadIDTxl()
        network_analysis = MultivariateTE()
        
        # Przekierowanie wyników do pliku logów
        sys.stdout = open("logs/logs_run_multivariate_te_shortened.txt", "w")
        
        # Parametry EEG
        srcDir = ''  # Upewnij się, że ścieżka jest poprawna
        subCode = 'RGA798'
        cond = 'art_watch2'
        samplingRate = 1000  # Hz
        samplesPerMs = samplingRate / 1000

        # Wczytanie EEG
        eeg = loadRawEEG(srcDir, subCode, cond)
        events, event_id = mne.events_from_annotations(eeg)
        
        # Wybór markerów "Response/P" i "Response/M"
        p_codes = [code for key, code in event_id.items() if key.startswith("Response/P")]
        m_codes = [code for key, code in event_id.items() if key.startswith("Response/M")]
        
        if not p_codes or not m_codes:
            raise ValueError("Nie znaleziono markerów 'Response/P' lub 'Response/M'.")
        
        p_events = events[np.isin(events[:, 2], p_codes)][:, 0]
        m_events = events[np.isin(events[:, 2], m_codes)][:, 0]
        
        print("Markery P:", p_events, flush=True)
        print("Markery M:", m_events, flush=True)
        
        # Tworzenie epok
        epoch_list = []
        fs = eeg.info["sfreq"]
        for p_time in p_events:
            m_time = m_events[m_events > p_time]
            if len(m_time) == 0:
                continue
            m_time = m_time[0]
            try:
                epoch = eeg.copy().crop(tmin=p_time / fs, tmax=m_time / fs)
                epoch_list.append(epoch)
            except Exception as e:
                print(f"Błąd przy tworzeniu epoki od {p_time/fs:.3f}s do {m_time/fs:.3f}s: {e}", flush=True)
        
        print(f"Znaleziono {len(epoch_list)} epok EEG.", flush=True)
        
        # Skrócone ustawienia analizy TE
        settings = {
        'cmi_estimator': 'JidtGaussianCMI',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_omnibus': 21,
        'n_perm_max_seq': 21,
        'max_lag_sources': 300,  # maksymalnie 300 ms
        'min_lag_sources': int(20 * samplesPerMs),  # minimalnie 20 ms
        "alpha_min_stat": 0.05,
        "alpha_max_stat": 0.05,
        "alpha_omnibus": 0.05,
        "alpha_max_seq": 0.05,
        "pastSpan": 300,  # zakres wstecz 300 ms
        "step": 100,      # krok co 100 ms
        "verbose": True
        }

        settings_fast_demo = {
        'cmi_estimator': 'JidtGaussianCMI',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_omnibus': 21,
        'n_perm_max_seq': 21,
        'max_lag_sources': 50,    # 50 ms – szybkie obliczenia
        'min_lag_sources': 10,    # 10 ms
        "alpha_min_stat": 0.05,
        "alpha_max_stat": 0.05,
        "alpha_omnibus": 0.05,
        "alpha_max_seq": 0.05,
        'pastSpan': 100,
        'step': 100,
        'verbose': True
        }

        # Wybór 10 elektrod jako źródła (indeksy 0-9) i 10 jako cele (indeksy 22-31)
        sources = [0,1]# 2, 3, 4, 5, 6, 7, 8, 9]
        targets = [22,23]#, 24, 25, 26, 27, 28, 29, 30, 31]
        
        results_list = []
        
        # Analiza TE dla każdej epoki
        for i, epoch in enumerate(epoch_list):
            print(f"Analizuję epokę {i+1}/{len(epoch_list)}", flush=True)
            data_array = epoch.get_data() * 1e6  # skalowanie do µV
            data = Data(data_array, dim_order='ps', normalise=True, seed=1)
            try:
                results = network_analysis.analyse_network(
                    settings=settings_fast_demo,
                    data=data,
                    sources=sources,
                    targets=targets
                )
                print(f"Epoka {i+1} zakończona.", flush=True)
                results_list.append(results)
            except Exception as e:
                print(f"Błąd podczas analizy TE w epoce {i+1}: {e}", flush=True)
                continue
            
            try:
                print(f"Wyniki epoki {i+1}:", flush=True)
                results.print_edge_list(weights='max_te_lag', fdr=False)
            except Exception as e:
                print(f"Błąd przy wypisywaniu wyników epoki {i+1}: {e}", flush=True)
        
        # Analiza statystyk lagów
        past_spans = []
        for i, results in enumerate(results_list):
            try:
                edges = results.print_edge_list(weights='max_te_lag', fdr=False)
                lags = [edge[2] for edge in edges]
                past_spans.extend(lags)
            except Exception as e:
                print(f"Błąd przy analizie statystyk epoki {i+1}: {e}", flush=True)
        
        if past_spans:
            past_spans = np.array(past_spans)
            print(f"Min pastSpan: {np.min(past_spans)} ms", flush=True)
            print(f"Max pastSpan: {np.max(past_spans)} ms", flush=True)
            print(f"Średni pastSpan: {np.mean(past_spans):.2f} ms", flush=True)
            print(f"Mediana pastSpan: {np.median(past_spans):.2f} ms", flush=True)
        else:
            print("Brak wyników do statystyk lagów.", flush=True)
        
        # Zapis wykresu sieci dla ostatniej epoki
        if results_list:
            try:
                plot_network(results=results_list[-1], weights='max_te_lag', fdr=False)
                plt.savefig("plots/network_last_epoch_shortened.png")
                plt.close()
            except Exception as e:
                print(f"Błąd przy zapisywaniu wykresu sieci: {e}", flush=True)
        
        print("Analiza zakończona. Wyniki zapisane do plików.", flush=True)
    
    except Exception as e:
        print(f"Krytyczny błąd: {e}", flush=True)

if __name__ == '__main__':
    main()
