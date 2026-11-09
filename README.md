# README

---

# Deep\_PPG



基于 DeepPPG 模型的 PPG 信号处理与 LOSO\-CV \(留一验证\) 训练项目



## 项目结构

```Plain Text
Deep_PPG/
|--dataset/
|    |--PPG_FieldStudy/ # Raw dataset
|
|--processed_data/
|    |--dalia_fft/
|    |--dalia_segments/ # processed data 
|
|--scripts/
|    |--preprocess_fft.py
|    |--preprocess_npz.py
|    |--run_loocv.py
|    |--verify_fft.py 
|
|--src/
|    |--data_processing/
|            |--fft_spectrum.py
|            |--load_to_npz.py 
|  
|    |--model/
|        |--model.py
|
|    |--train/
|        |--dataset.py
|        |--loocv.py
|        |--train_utils.py
|    |--checkpoints/
|        |-- fold_1_best.pth    # subject 1 当测试集时的最佳权重
|        |-- fold_2_best.pth
|        | ...
|        |-- fold_15_best.pth
|        |-- summary.pth   
|
|--.gitignore
|--README.md
```

## 环境依赖



- Python \>= 3\.8

- PyTorch \(支持 CUDA\)

- NumPy, SciPy

- 安装依赖：`pip install -r requirements.txt`

    

## 数据准备



本项目**不包含**原始数据与预处理后的 `.npz` 文件（已通过 `.gitignore` 屏蔽）。



1. 下载原始数据集（如 PPG FieldStudy / Dalia），放置于 `dataset/PPG_FieldStudy/`。

2. 运行预处理脚本生成数据：

```Bash
python -m scripts.preprocess_npz
python -m scripts.preprocess_fft
```

- 源代码未修改路径，需要先修改路径

- 处理后的数据默认存在 Deep\_PPG/processed\_data/dalia\_segments/ 和 Deep\_PPG/processed\_data/dalia\_fft/下



## 训练

```Bash
python -m scripts.run_loocv
```

- 可以在后面自定义参数，参数详见 run\_loocv\.py

```Plain Text
参数说明：
- `--data_dir`: 预处理数据路径（默认 `processed_data/dalia_fft`）
- `--num_subjects`: 受试者数量（默认 15）
- `--num_epochs`: 训练轮次（默认 100）
- `--N_tr`: 模型时间窗口参数（默认 7）
- `--device`: `cuda` 或 `cpu`
-  --save_dir: 模型权重的保存位置

## 训练逻辑

- **LOSO-CV**：15 个受试者，每次留 1 个作测试，其余 14 个训练，共 15 个 Fold。
- **模型**：DeepPPG，损失函数采用 L1Loss（MAE）。
- **输出**：每个 Fold 的最佳测试 MAE，最终输出平均 MAE。

## 开源与引用

参考原始数据集说明：`dataset/PPG_FieldStudy/PPG_FieldStudy_readme.pdf`
参考Deep_PPG原始论文：Reiss, A., Indlekofer, I., Schmidt, P., & Van Laerhoven, K. (2019). Deep PPG: Large-Scale Heart Rate Estimation with Convolutional Neural Networks. Sensors, 19(14), 3079. https://doi.org/10.3390/s19143079
```



### 保存权重

```Bash
python -m scripts.run_loocv --save_dir "D:/path/to/your/checkpoints"
```

不传 `--save_dir` 时默认保存到项目根目录下的 `checkpoints/`。



### 保存内容



每个 `fold_X_best.pth` 包含：



|字段|说明|
|---|---|
|`model_state_dict`|模型权重参数|
|`optimizer_state_dict`|优化器状态（支持断点续训）|
|`best_test_mae`|该 fold 最佳测试 MAE|
|`epoch`|达到最佳 MAE 的 epoch 数|
|`test_subject`|该 fold 对应的测试受试者编号|



`summary.pth` 包含：



|字段|说明|
|---|---|
|`all_maes`|每个 fold 的最佳 MAE 列表|
|`mean_mae`|所有 fold 的平均 MAE|
|`config`|本次训练的配置参数|







