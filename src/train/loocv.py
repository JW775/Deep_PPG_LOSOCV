# 加载当前 fold 的 train/test 数据，构建 DataLoader
# 初始化模型、optimizer、criterion
# 跑 num_epochs 个 epoch（每个 epoch 调 train_epoch + eval_epoch）
# 记录每轮测试集的 MAE
# 15 轮跑完，汇总所有 subject 的测试结果

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os

ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.append(os.path.abspath(ROOT_DIR))

from src.model.model import DeepPPG
from dataset import PPGDaliaDataset
from train_utils import train_epoch, eval_epoch

def run_losocv(
        data_dir,
        num_subjects = 15,
        batch_size = 32, 
        num_epochs = 100,
        N_tr = 7,
        lr = 1e-3,
        device = 'cuda',
        save_dir = None,    
):

    all_maes = []

    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        device = 'cpu'

    device = torch.device(device)    
    print(f"using device: {device}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for test_subject in range(1, num_subjects + 1):
        print(f"Fold: {test_subject}/{num_subjects}")

        train_subjects = []
        for i in range(1, num_subjects + 1):
            if i == test_subject:
                continue
            train_subjects.append(i)

        train_dataset = PPGDaliaDataset(train_subjects, data_dir, N_tr = N_tr)
        test_dataset = PPGDaliaDataset([test_subject], data_dir, N_tr = N_tr)

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last= False, num_workers=4, pin_memory=True,)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last=False)

        # 初始化模型
        model = DeepPPG().to(device)
        optimizer = optim.Adam(model.parameters(), lr = lr)
        criterion = nn.L1Loss()

        # 训练
        best_test_mae = float('inf')
        for epoch in range(1, num_epochs + 1):
            train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
            test_mae = eval_epoch(model, test_loader, criterion, device)

            if test_mae < best_test_mae:
                best_test_mae = test_mae
                if save_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_test_mae': best_test_mae,
                        'test_subject': test_subject,
                    }, os.path.join(save_dir, f'fold_{test_subject}_best.pth'))

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch: {epoch}/{num_epochs}, Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

        all_maes.append(best_test_mae)

    print(f"Mean MAE: {sum(all_maes) / len(all_maes):.4f}")
    for i, mae in enumerate(all_maes):
        print(f"Subject {i+1}: {mae:.4f}")
    if save_dir is not None:
        torch.save({
            'all_maes': all_maes,
            'mean_mae': mean_mae,
            'config': {
                'num_subjects': num_subjects,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'N_tr': N_tr,
                'lr': lr,
            },
        }, os.path.join(save_dir, 'summary.pth'))

    return all_maes
        

    

# 临时测试
if __name__ == "__main__":
    run_losocv(
        data_dir="D:/develop/IDEA/PPG_project/Deep_PPG/processed_data/dalia_fft",
        num_subjects=15,
        batch_size = 32, 
        num_epochs=3, 
        N_tr = 7,
        lr = 1e-3,
        device = 'cuda',
        save_dir = "D:/develop/IDEA/PPG_project/Deep_PPG/checkpoints",
    )