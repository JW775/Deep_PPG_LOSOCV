"""
LOSO-CV 训练入口脚本
用法:
    python -m src.train.run
    python -m src.train.run --num_epochs 100 --batch_size 32
"""
import argparse
import os
import sys

import torch

# 确保项目根目录在 sys.path 里
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.train.loocv import run_losocv

# 默认配置
DEFAULT_CONFIG = {
    "data_dir": "D:/develop/IDEA/PPG_project/Deep_PPG/processed_data/dalia_fft",
    "num_subjects": 15,
    "batch_size": 32,
    "num_epochs": 100,
    "N_tr": 7,
    "lr": 1e-3,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "save_dir": "D:/develop/IDEA/PPG_project/Deep_PPG/checkpoints", 
}


def parse_args():
    parser = argparse.ArgumentParser(description="DeepPPG LOSO-CV Training")

    parser.add_argument("--data_dir", type=str, default=DEFAULT_CONFIG["data_dir"])
    parser.add_argument("--num_subjects", type=int, default=DEFAULT_CONFIG["num_subjects"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--num_epochs", type=int, default=DEFAULT_CONFIG["num_epochs"])
    parser.add_argument("--N_tr", type=int, default=DEFAULT_CONFIG["N_tr"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--device", type=str, default=DEFAULT_CONFIG["device"])
    parser.add_argument("--save_dir", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DeepPPG LOSO-CV Training")
    print("=" * 60)
    print(f"  data_dir     : {args.data_dir}")
    print(f"  num_subjects : {args.num_subjects}")
    print(f"  batch_size   : {args.batch_size}")
    print(f"  num_epochs   : {args.num_epochs}")
    print(f"  N_tr         : {args.N_tr}")
    print(f"  lr           : {args.lr}")
    print(f"  device       : {args.device}")
    print("=" * 60)

    run_losocv(
        data_dir=args.data_dir,
        num_subjects=args.num_subjects,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        N_tr=args.N_tr,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()