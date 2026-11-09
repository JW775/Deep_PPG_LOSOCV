import sys
import os
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.fft_spectrum import fft_and_normalize

NPZ_DIR = 'D:/develop/IDEA/PPG_project/Deep_PPG/processed_data/dalia_segments'
FFT_OUT_DIR = 'D:/develop/IDEA/PPG_project/Deep_PPG/processed_data/dalia_fft'

fs_ppg = 64
fs_acc = 32
target_len = 1025

def process_one_subject(npz_path, fft_out_dir):
    data = np.load(npz_path, allow_pickle=True)

    ppg = data['ppg']
    acc_0 = data['acc_0']
    acc_1 = data['acc_1']
    acc_2 = data['acc_2']
    labels = data['label']

    N = len(ppg)
    X = np.zeros((N, 4, target_len))
    for i in range(N):
        X[i, 0, :] = fft_and_normalize(ppg[i], target_len, fs_ppg)
        X[i, 1, :] = fft_and_normalize(acc_0[i], target_len, fs_acc)
        X[i, 2, :] = fft_and_normalize(acc_1[i], target_len, fs_acc)
        X[i, 3, :] = fft_and_normalize(acc_2[i], target_len, fs_acc)

        if (i+1) % 500 == 0:
            print(f'Processed {i+1} segments')

    subject_id = os.path.basename(npz_path).replace('s_', '').replace('.npz', '')
    save_path = os.path.join(fft_out_dir, f's_{subject_id}.npz')
    np.savez(save_path, X=X, y = labels)
    print(f'Saved to {save_path}')
    print(f"X shape: {X.shape}")
    print(f"y shape: {labels.shape}")


if __name__ == '__main__':
    os.makedirs(FFT_OUT_DIR, exist_ok=True)

    for fname in sorted(os.listdir(NPZ_DIR)):
        if not fname.endswith('.npz'):
            continue
        print(f"processing {fname}")
        process_one_subject(os.path.join(NPZ_DIR, fname), FFT_OUT_DIR)

    print("Done")