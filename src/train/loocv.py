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
import random

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
        save_dir = "./checkpoints/PPG_LOSO_bs32_lr1e-3" , 
        num_val_subjects = 2, # 每个fold抽取2个受试者
        seed = 42             # 确保划分可复现   
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

        
        candidates = []
        for i in range(1, num_subjects + 1):
            if i == test_subject:
                continue
            candidates.append(i)

        
        rng = random.Random(seed + test_subject)
        val_subjects = rng.sample(candidates, num_val_subjects)
        train_subjects = [s for s in candidates if s not in val_subjects]

        print(f"train subjects: {train_subjects}")
        print(f"test subjects: {test_subject}")
        train_dataset = PPGDaliaDataset(train_subjects, data_dir, N_tr = N_tr)
        val_dataset = PPGDaliaDataset(val_subjects, data_dir, N_tr = N_tr)
        test_dataset = PPGDaliaDataset([test_subject], data_dir, N_tr = N_tr)

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last= False, num_workers=4, pin_memory=True,)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, drop_last=False)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, drop_last=False)

        # 初始化模型
        model = DeepPPG().to(device)
        optimizer = optim.Adam(model.parameters(), lr = lr)
        criterion = nn.L1Loss()

        # 训练
        best_val_mae = float('inf')
        best_epoch = 0
        best_state_dict = None
        for epoch in range(1, num_epochs + 1):
            train_mae = train_epoch(model, train_loader, optimizer, criterion, device)
            val_mae = eval_epoch(model, val_loader, criterion, device)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch = epoch
            
                best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
                if save_dir is not None:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_mae': val_mae,
                        'test_subject': test_subject,
                        'val_subjects': val_subjects,
                        'config': {'N_tr': N_tr, 'lr': lr, 'batch_size': batch_size},
                    }, os.path.join(save_dir, f'fold_{test_subject}_best.pth'))

            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{num_epochs}, train MAE: {train_mae:.4f}, val MAE: {val_mae:.4f}")
        
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        test_mae = eval_epoch(model, test_loader, criterion, device)
        print(f"fold {test_subject}: Test MAE: {test_mae:.4f}" 
              f"(best Val MAE: {best_val_mae:.4f} at epoch {best_epoch})")
        all_maes.append(test_mae)

    mean_mae = sum(all_maes) / len(all_maes)
    print(f"Mean Test MAE: {mean_mae:.4f}")
    for i, mae in enumerate(all_maes):
        print(f"subject{i+1}: {mae:.4f}")

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
                'num_val_subjects': num_val_subjects,
                'seed': seed,
            },
        }, os.path.join(save_dir, 'summary.pth'))

    return all_maes    
        


            

        