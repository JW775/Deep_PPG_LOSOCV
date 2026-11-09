import sys
import os 





sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.load_to_npz import load_ppg_dalia_files_to_npz

DATASET_PATH = 'D:/develop/IDEA/PPG_project/Deep_PPG/dataset/PPG_FieldStudy'
SAVED_PATH = 'D:/develop/IDEA/PPG_project/Deep_PPG/processed_data/dalia_segments'

if __name__ == "__main__":
    load_ppg_dalia_files_to_npz(DATASET_PATH, SAVED_PATH)