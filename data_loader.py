from scaler_manager import ScalerManager
import pandas as pd
import torch
import time

def load_csv_data(file_path, device='cpu'):
    df = pd.read_csv(file_path)

    # 时间转换为秒数
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()

    # 初始化 ScalerManager 并拟合
    scaler_mgr = ScalerManager()
    scaler_mgr.fit(df)

    # 归一化输入特征
    features_norm = scaler_mgr.transform_all(df)
    inputs = torch.tensor(features_norm, dtype=torch.float32).to(device)

    # 提取目标速度
    targets = df[['uo', 'vo']].values
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    # 拆分归一化特征
    t_norm = inputs[:, 0:1]
    x_norm = inputs[:, 1:2]
    y_norm = inputs[:, 2:3]
    z_norm = inputs[:, 3:4]
    u_true = targets[:, 0:1]
    v_true = targets[:, 1:2]

    return t_norm, x_norm, y_norm, z_norm, u_true, v_true, scaler_mgr, df

""""#def split_dataset_by_time(csv_path, train_ratio=0.8):
    df = pd.read_csv(csv_path)

    # 时间转换为秒数（与 load_csv_data 保持一致）
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()

    # 按时间排序
    df_sorted = df.sort_values(by='time').reset_index(drop=True)

    # 按比例划分
    split_index = int(len(df_sorted) * train_ratio)
    train_df = df_sorted.iloc[:split_index].copy()
    test_df = df_sorted.iloc[split_index:].copy()

    print(f"✅ 数据划分完成：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")
    print(f"📊 时间范围：训练集 time ∈ [{train_df['time'].min()}, {train_df['time'].max()}]")
    print(f"📊 时间范围：测试集 time ∈ [{test_df['time'].min()}, {test_df['time'].max()}]")

    return train_df, test_df
"""
def split_dataset_random(csv_path, train_ratio=0.8, seed=None):
    if seed is None:
        seed = int(time.time()*1000%2**32)


    df = pd.read_csv(csv_path)

    # 时间转换为秒数（保持一致）
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = (df['time'] - df['time'].min()).dt.total_seconds()

    # 随机打乱
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # 按比例划分
    split_index = int(len(df_shuffled) * train_ratio)
    train_df = df_shuffled.iloc[:split_index].copy()
    test_df = df_shuffled.iloc[split_index:].copy()

    print(f"✅ 随机划分完成：训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")
    return train_df, test_df


def load_csv_data_from_df(df, device='cpu'):
    from scaler_manager import ScalerManager
    import torch

    # 时间列已是秒数，无需转换
    scaler_mgr = ScalerManager()
    scaler_mgr.fit(df)

    # 归一化输入特征
    features_norm = scaler_mgr.transform_all(df)
    inputs = torch.tensor(features_norm, dtype=torch.float32).to(device)

    # 提取目标速度
    targets = df[['uo', 'vo']].values
    targets = torch.tensor(targets, dtype=torch.float32).to(device)

    # 拆分归一化特征
    t_norm = inputs[:, 0:1]
    x_norm = inputs[:, 1:2]
    y_norm = inputs[:, 2:3]
    z_norm = inputs[:, 3:4]
    u_true = targets[:, 0:1]
    v_true = targets[:, 1:2]

    return t_norm, x_norm, y_norm, z_norm, u_true, v_true, scaler_mgr, df
