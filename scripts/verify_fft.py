# scripts/verify_fft.py
import numpy as np
import matplotlib.pyplot as plt

data = np.load('D:/develop/IDEA/PPG_project/Deep_PPG/processed_data/dalia_fft/s_1.npz')

X = data['X']   # (4603, 4, 1025)
y = data['y']   # (4603,)

print(f"X shape: {X.shape}, y shape: {y.shape}")
print(f"X range: [{X.min():.3f}, {X.max():.3f}]")  # z-score 后应该在 ±5 左右
print(f"y range: [{y.min():.1f}, {y.max():.1f}] bpm")  # 应该大约 50–180 bpm

# 画前 3 个窗口的 PPG 频谱
fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i in range(3):
    axes[i].plot(X[i, 0, :])
    axes[i].set_title(f'Window {i}, Label: {y[i]:.0f} bpm')
    axes[i].set_xlabel('Frequency bin (0–4 Hz)')
    axes[i].set_ylabel('Magnitude (z-scored)')
plt.tight_layout()
plt.show()

# 转成 Hz 看峰值
freq_bins = np.linspace(0, 4, 1025)
bpm_bins = freq_bins * 60
for i in range(3):
    peak_idx = np.argmax(X[i, 0, :])
    print(f"Window {i}: peak at {freq_bins[peak_idx]:.2f} Hz = {bpm_bins[peak_idx]:.0f} bpm, label = {y[i]:.0f} bpm")