import numpy as np
import torch
from torch.utils.data import Dataset
import os

class PPGDaliaDataset(Dataset):
    def __init__(self, subject_ids, data_dir, N_tr=7, mode='train'):
        self.N_tr = N_tr
        self.mode = mode

        # 文件级缓存：每个 .npz 只读一次
        self.file_cache = {}
        self.samples = []

        for sid in subject_ids:
            file_path = os.path.join(data_dir, f's_{sid}.npz')

            if file_path not in self.file_cache:
                with np.load(file_path) as f:
                    X = f['X'].astype(np.float32)
                    y = f['y'].astype(np.float32)
                self.file_cache[file_path] = (X, y)

            X, y = self.file_cache[file_path]

            if X.ndim == 3:
                M = X.shape[0] - N_tr + 1
                for i in range(M):
                    self.samples.append((file_path, i, i + N_tr, y[N_tr - 1 + i]))
            elif X.ndim == 4 and X.shape[2] == N_tr:
                for i in range(X.shape[0]):
                    self.samples.append((file_path, i, None, y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, start, end, label = self.samples[idx]
        X, _ = self.file_cache[file_path]

        if end is not None:
            window = X[start:end, :, :]
            window = np.transpose(window, (1, 0, 2))
        else:
            window = X[start]

        return (
            torch.from_numpy(window),
            torch.tensor(label, dtype=torch.float32).view(1),
        )