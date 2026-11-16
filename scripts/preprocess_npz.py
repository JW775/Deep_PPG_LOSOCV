import sys
import os 





sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing.load_to_npz import load_ppg_dalia_files_to_npz

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(ROOT_DIR, 'dataset', 'PPG_FieldStudy')
SAVED_PATH   = os.path.join(ROOT_DIR, 'processed_data', 'dalia_segments')

if __name__ == "__main__":
    load_ppg_dalia_files_to_npz(DATASET_PATH, SAVED_PATH)