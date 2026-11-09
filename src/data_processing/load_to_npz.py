import os
import pickle
import numpy as np
import pandas as pd




def load_ppg_dalia_files_to_npz(dataset_path, save_dir,fs_ppg=64, fs_acc=32, win_len=8, win_step=2):
    os.makedirs(save_dir, exist_ok=True)

    for subject_floder in os.listdir(dataset_path):
        subject_path = os.path.join(dataset_path, subject_floder)

        if os.path.isdir(subject_path):
            for filename in os.listdir(subject_path):
                if filename.endswith('pkl'):
                    file_path = os.path.join(subject_path, filename)

                    with open(file_path, 'rb') as f:
                        data = pickle.load(f, encoding='latin1')

                    subject_id = int(subject_floder[1:])  # 提取 subject_id
                    ppg_raw = data['signal']['wrist']['BVP']
                    acc_raw_0 = data['signal']['wrist']['ACC'][:,0]
                    acc_raw_1 = data['signal']['wrist']['ACC'][:,1]
                    acc_raw_2 = data['signal']['wrist']['ACC'][:,2]
                    ppg_segments = []
                    acc_segments_0 = []
                    acc_segments_1 = []
                    acc_segments_2 = []
                    for i in range(0, len(ppg_raw) - win_len * fs_ppg + 1, win_step * fs_ppg):
                        ppg_segment = ppg_raw[i:i + win_len * fs_ppg]
                        ppg_segments.append(ppg_segment)

                    for i in range(0, len(acc_raw_0) - win_len * fs_acc + 1, win_step * fs_acc):
                        acc_segment_0 = acc_raw_0[i:i + win_len * fs_acc]
                        acc_segment_1 = acc_raw_1[i:i + win_len * fs_acc]
                        acc_segment_2 = acc_raw_2[i:i + win_len * fs_acc]
                        acc_segments_0.append(acc_segment_0)
                        acc_segments_1.append(acc_segment_1)
                        acc_segments_2.append(acc_segment_2)

                    labels = np.array(data['label'])
                    ppg_segments = np.array(ppg_segments)
                    acc_segments_0 = np.array(acc_segments_0)
                    acc_segments_1 = np.array(acc_segments_1)
                    acc_segments_2 = np.array(acc_segments_2)

                    # 对齐长度
                    min_len = min(len(ppg_segments), len(acc_segments_0), len(labels))
                    ppg_segments = ppg_segments[:min_len]
                    acc_segments_0 = acc_segments_0[:min_len]
                    acc_segments_1 = acc_segments_1[:min_len]
                    acc_segments_2 = acc_segments_2[:min_len]
                    labels = labels[:min_len]

                    # 保存npz文件
                    save_path_npz = os.path.join(save_dir, f's_{subject_id}.npz')
                    np.savez(
                        save_path_npz,
                        ppg=ppg_segments,
                        acc_0=acc_segments_0,
                        acc_1=acc_segments_1,
                        acc_2=acc_segments_2,
                        label=labels
                    )

                    print(f"S{subject_id}: PPG {ppg_segments.shape}, "
                        f"ACC {acc_segments_0.shape}, Labels {labels.shape}")

    print("All files have been processed and saved to npz format.")
            


