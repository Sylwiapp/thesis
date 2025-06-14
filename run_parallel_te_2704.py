#!/usr/bin/env python3
import os, time
import numpy as np
import matplotlib.pyplot as plt
import mne
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
from idtxl.visualise_graph import plot_network
from toolkit2 import setCwdHere, loadIDTxl, loadRawEEG, plotSingleTargetMteTimeSeries

# -------------------------------------------------------------------
# 1) Parametry TE i ścieżki
# -------------------------------------------------------------------
settings = {
    'cmi_estimator':   'JidtGaussianCMI',
    'n_perm_max_stat':   21,
    'n_perm_min_stat':   21,
    'n_perm_omnibus':    21,
    'n_perm_max_seq':    21,
    'max_lag_sources':   50,
    'min_lag_sources':   10,
    'alpha_min_stat':   0.5,
    'alpha_max_stat':   0.5,
    'alpha_omnibus':    0.5,
    'alpha_max_seq':    0.5,
    'pastSpan':         100,
    'step':            2000,
    'verbose':         True,
    'fdr_correction': False
}
sources = [0, 1]
targets = [22, 23]

# katalog na wykresy
os.makedirs('plots', exist_ok=True)

# -------------------------------------------------------------------
# 2) Inicjalizacja IDTxl
# -------------------------------------------------------------------
setCwdHere()
loadIDTxl()
mte = MultivariateTE()

# -------------------------------------------------------------------
# 3) Wczytaj EEG i wytnij epoki
# -------------------------------------------------------------------
eeg = loadRawEEG('', 'RGA798', 'art_watch2')
events, event_id = mne.events_from_annotations(eeg)
p_codes = [c for k,c in event_id.items() if k.startswith("Response/P")]
m_codes = [c for k,c in event_id.items() if k.startswith("Response/M")]
p_events = events[np.isin(events[:,2], p_codes)][:,0]
m_events = events[np.isin(events[:,2], m_codes)][:,0]

fs = eeg.info['sfreq']
epoch_list = []
for p in p_events:
    m_after = m_events[m_events>p]
    if len(m_after)==0: continue
    epoch_list.append(eeg.copy().crop(tmin=p/fs, tmax=m_after[0]/fs))

print(f"Znalazłem {len(epoch_list)} epok.")
if len(epoch_list)<2:
    raise RuntimeError("Potrzebujesz co najmniej 2 epoki do testu.")

# -------------------------------------------------------------------
# 4) Wybieramy DRUGĄ epokę (indeks 1)
# -------------------------------------------------------------------
epoch = epoch_list[2]
print("Analiza DRUGIEJ epoki (indeks 1) …")
t0 = time.time()

# przygotowanie danych do IDTxl
data_array = epoch.get_data() * 1e6
data = Data(data_array, dim_order='ps', normalise=True, seed=1)

# analiza TE
res = mte.analyse_network(
    settings=settings,
    data=data,
    sources=sources,
    targets=targets
)
print(f"Czas obliczeń: {time.time()-t0:.1f}s\n")

# -------------------------------------------------------------------
# 5) Wypisz krawędzie
# -------------------------------------------------------------------
print("===== EDGE LIST =====")
res.print_edge_list(weights='max_te_lag', fdr=False)

# -------------------------------------------------------------------
# 6) Adjacency matrix → obraz (interaktywnie, bez zapisu)
# -------------------------------------------------------------------
adj = res.get_adjacency_matrix(weights='max_te_lag', fdr=False)

# konwersja do numpy
try:
    arr = adj.toarray()
except AttributeError:
    arr = np.array(adj)

# włącz interaktywny tryb
plt.ion()

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    arr,
    cmap='viridis',
    interpolation='nearest',
    aspect='equal',
    origin='lower'
)
ax.set_title('Adjacency (max_te_lag)')
ax.set_xlabel('Target #')
ax.set_ylabel('Source #')

# dodaj pasek kolorów
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Lag [samples]')

plt.tight_layout()
plt.show()

# -------------------------------------------------------------------
# 7) Graf sieci → obraz
# -------------------------------------------------------------------
graph, fig2 = plot_network(res, weights='max_te_lag', fdr=False)
fig2.set_size_inches(6,6)
fig2.savefig('plots/epoch2_network.png')
plt.show()
plt.close(fig2)
print("Zapisano: plots/epoch2_network.png")


# -------------------------------------------------------------------
# 8) Histogram lagów → obraz
# -------------------------------------------------------------------
# zbieramy wszystkie niezerowe lagi
src_idx, tgt_idx = np.nonzero(arr)
lags = arr[src_idx, tgt_idx]

fig3 = plt.figure(figsize=(6,4))
plt.hist(lags, bins=20, edgecolor='black')
plt.title('Histogram TE-lagów')
plt.xlabel('Lag [samples]')
plt.ylabel('Ilość krawędzi')
plt.grid(True, linestyle='--', alpha=0.5)
fig3.savefig('plots/epoch2_lag_histogram.png')
plt.close(fig3)
print("Zapisano: plots/epoch2_lag_histogram.png")

# -------------------------------------------------------------------
# 9) TE-time series dla każdego targetu → obrazy
# -------------------------------------------------------------------
for tgt in targets:
    fig4 = plt.figure(figsize=(8,3))
    # funkcja z toolkit2 przyjmuje listę wyników i numer targetu:
    plotSingleTargetMteTimeSeries([res], tgt)
    plt.title(f"TE time-series dla target={tgt}")
    fig4.savefig(f'plots/epoch2_target{tgt}_timeseries.png')
    plt.close(fig4)
    print(f"Zapisano: plots/epoch2_target{tgt}_timeseries.png")

print("\n=== Gotowe! Wszystkie wykresy w katalogu plots/ ===")
