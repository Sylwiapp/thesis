from mpi4py import MPI
import sys
import numpy as np
import matplotlib.pyplot as plt
import mne
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network
from toolkit2 import setCwdHere, loadIDTxl, loadRawEEG
import torch
import time


def split_list(data, n):
    """Dzieli listę data na n części możliwie równomiernie."""
    k, m = divmod(len(data), n)
    return [data[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start_time = time.time()

    sys.stdout = open(f"logs/log_rank_{rank}.txt", "w")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}', flush=True)

    setCwdHere()
    loadIDTxl()
    network_analysis = MultivariateTE()

    srcDir = ''
    subCode = 'RGA798'
    cond = 'art_watch2'
    samplingRate = 1000
    samplesPerMs = samplingRate / 1000

    eeg = loadRawEEG(srcDir, subCode, cond)
    events, event_id = mne.events_from_annotations(eeg)

    p_codes = [code for key, code in event_id.items() if key.startswith("Response/P")]
    m_codes = [code for key, code in event_id.items() if key.startswith("Response/M")]

    if not p_codes or not m_codes:
        raise ValueError("Markers 'Response/P' or 'Response/M' not found.")

    p_events = events[np.isin(events[:, 2], p_codes)][:, 0]
    m_events = events[np.isin(events[:, 2], m_codes)][:, 0]

    epoch_list = []
    fs = eeg.info["sfreq"]
    for p_time in p_events:
        m_time = m_events[m_events > p_time]
        if len(m_time) == 0:
            continue
        m_time = m_time[0]
        epoch = eeg.copy().crop(tmin=p_time / fs, tmax=m_time / fs)
        epoch_list.append(epoch)

    epoch_list = epoch_list[:1] 
    epoch_list_split = split_list(epoch_list, size)
    my_epochs = epoch_list_split[rank]

    settings_fast_demo = {
        'cmi_estimator': 'JidtGaussianCMI',
        'n_perm_max_stat': 21,
        'n_perm_min_stat': 21,
        'n_perm_omnibus': 21,
        'n_perm_max_seq': 21,
        'max_lag_sources': 50,
        'min_lag_sources': int(10 * samplesPerMs),
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

    results_list = []
    epoch_list = epoch_list[:1]  # tylko 2
    
    for i, epoch in enumerate(my_epochs):
        print(f"Rank {rank} analizuje epokę {i+1}/{len(my_epochs)}", flush=True)
        data_array = epoch.get_data() * 1e6
        data = Data(data_array, dim_order='ps', normalise=True, seed=1)
        results = network_analysis.analyse_network(
            settings=settings_fast_demo,
            data=data,
            sources=sources,
            targets=targets
        )
        results_list.append(results)
        results.print_edge_list(weights='max_te_lag', fdr=False)

    # Zbieranie wyników z wszystkich wątków
    all_results = comm.gather(results_list, root=0)

    if rank == 0:
        total_time = time.time() - start_time
        print(f"Całkowity czas: {total_time:.2f} s", flush=True)

        combined_edges = []

        for thread_results in all_results:
            for res in thread_results:
                # lagi dla każdego polaczenia
                adj_matrix_lag = np.asarray(res.get_adjacency_matrix(weights='max_te_lag', fdr=False))
                #binarne info - polaczenie czy nie?
                adj_matrix_binary = np.asarray(res.get_adjacency_matrix(weights='binary', fdr=False))

                print(type(res.get_adjacency_matrix(weights='binary', fdr=False)))
                sources, targets = np.nonzero(adj_matrix_binary)
                binary_links = adj_matrix_binary.data
                lags = adj_matrix_lag[sources, targets]

                for src, tgt, lag in zip(sources, targets, lags):
                    combined_edges.append((src, tgt, lag))

        # Zapisz wyniki zbiorcze
        with open("logs/combined_results.txt", "w") as f:
            for edge in combined_edges:
                f.write(f"{edge}\n")

        # Wykres zależności ilości połączeń od laga
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
            print("Brak znaczących połączeń we wszystkich epokach.")



if __name__ == '__main__':
    main()
