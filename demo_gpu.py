import numpy as np
import cupy as cp
import torch
import time

# Generowanie sztucznego EEG (64 kanały, 7000 próbek)
np.random.seed(42)
eeg_data = np.random.randn(64, 7000) * 1e-6  # μV

### 1. NumPy (CPU) ###
def process_numpy(data):
    start = time.time()
    norm_data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
    end = time.time()
    return norm_data, end - start

### 2. CuPy (GPU) ###
def process_cupy(data):
    try:
        data_gpu = cp.asarray(data)
        cp.cuda.Device(0).synchronize()  # Synchronizacja przed pomiarem
        start = time.time()
        norm_data_gpu = (data_gpu - cp.mean(data_gpu, axis=1, keepdims=True)) / cp.std(data_gpu, axis=1, keepdims=True)
        cp.cuda.Device(0).synchronize()  # Synchronizacja po pomiarze
        end = time.time()
        return cp.asnumpy(norm_data_gpu), end - start
    except cp.cuda.runtime.CUDARuntimeError:
        print("Brak dostępnego GPU dla CuPy.")
        return None, None

### 3. PyTorch (GPU) ###
def process_torch(data):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_torch = torch.tensor(data, dtype=torch.float32, device=device)
    
    if device == "cuda":
        torch.cuda.synchronize()  # Synchronizacja przed pomiarem
    start = time.time()
    
    norm_data_torch = (data_torch - data_torch.mean(dim=1, keepdim=True)) / data_torch.std(dim=1, keepdim=True)
    
    if device == "cuda":
        torch.cuda.synchronize()  # Synchronizacja po pomiarze
    end = time.time()
    
    return norm_data_torch.cpu().numpy(), end - start

# Wykonanie testów
cpu_result, cpu_time = process_numpy(eeg_data)
gpu_cupy_result, gpu_cupy_time = process_cupy(eeg_data)
gpu_torch_result, gpu_torch_time = process_torch(eeg_data)

# Wyniki
print(f"NumPy (CPU): {cpu_time:.6f} s")

if gpu_cupy_time is not None:
    print(f"CuPy (GPU): {gpu_cupy_time:.6f} s")
else:
    print("CuPy (GPU) - brak obsługi.")

if gpu_torch_time is not None:
    print(f"PyTorch (GPU): {gpu_torch_time:.6f} s")
else:
    print("PyTorch (GPU) - brak obsługi.")
