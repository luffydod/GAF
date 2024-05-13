import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import json

def load_config(config_file):
    # set default config file
    if config_file is None:
        config_file = "./config/config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def metrics(Y_hat, Y):
    """
        Metric
        - MAE
        - RMSE
        - MAPE
    """
    # 创建一个掩码，标记非零元素为True
    mask = torch.ne(Y, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)

    # MAE
    mae = torch.abs(torch.sub(Y_hat, Y)).type(torch.float32)
    # RMSE
    rmse = mae ** 2
    # MAPE

    # ipdb.set_trace()
    Y = torch.where(Y==0, torch.tensor(1e-6), Y)
    mape = mae / Y
    # mean
    mae = torch.mean(mae)
    rmse = rmse * mask
    rmse = torch.sqrt(torch.mean(rmse))
    mape = mape * mask
    mape = torch.mean(mape)

    return mae, rmse, mape


def Seq2Instance(data, num_his, num_pred):
    """
        将时间序列数据 data 转换为模型训练所需的输入输出（滑动窗口构建）
        INPUT:
            data(num_step, num_vertex)
        OUTPUT:
            X(num_sample, num_his, num_vertex)
            Y(num_sample, num_pred, num_vertex)
    """
    num_step, num_vertex = data.shape
    num_sample = num_step - num_his - num_pred + 1
    X = torch.zeros(num_sample, num_his, num_vertex)
    Y = torch.zeros(num_sample, num_pred, num_vertex)
    for i in range(num_sample):
        X[i] = data[i:i+num_his]
        Y[i] = data[i+num_his:i+num_his+num_pred]
    return X,Y


def count_parameters(model):
    """
    打印模型中可训练参数的数量。

    参数：
        model: 模型对象。

    返回：
        可训练参数的总数量。
    """
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable parameters: {:,}'.format(parameters))
    return


def plot_train_val_loss(train_total_loss, val_total_loss, dataset_name="PeMS-BAY"):
    """
    绘制训练损失和验证损失的曲线，并保存到文件中。

    参数：
        train_total_loss:   训练损失列表。
        val_total_loss:     验证损失列表。

    返回：
        无（图像被保存到文件中）。
    """
    # 参数类型检查
    if not isinstance(train_total_loss, list) or not isinstance(val_total_loss, list):
        raise TypeError("train_total_loss and val_total_loss must be lists.")
    
    # 设置Seaborn样式
    sns.set(style="whitegrid")

    # 绘制图形
    plt.figure(figsize=(10, 5))
    sns.lineplot(x=range(1, len(train_total_loss) + 1), y=train_total_loss, color='b', marker='s', label='Train')
    sns.lineplot(x=range(1, len(val_total_loss) + 1), y=val_total_loss, color='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss & Validation loss')
    
    # 保存图像到文件
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"./img/plot/{dataset_name}_train_val_loss_{timestamp}.png"
    plt.savefig(file_name)
    
    # 清除图形
    plt.close()