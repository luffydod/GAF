import pandas as pd
import torch
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


def metric(prediction, ground_truth):
    """
        Metric
        - MAE
        - RMSE
        - MAPE
    """
    # 创建一个掩码，标记非零元素为True
    mask = torch.ne(ground_truth, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)

    # MAE
    mae = torch.abs(torch.sub(prediction, ground_truth)).type(torch.float32)
    # RMSE
    rmse = mae ** 2
    # MAPE

    # ipdb.set_trace()
    ground_truth = torch.where(ground_truth==0, torch.tensor(1e-6), ground_truth)
    mape = mae / ground_truth
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


def load_SE(file_path):
    with open(file_path, mode='r') as f:
        lines = f.readlines()
        temp = lines[0].split(' ')
        # V=325,D=64
        num_vertex, dims = int(temp[0]), int(temp[1])

        SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
        for line in lines[1:]:
            temp = line.split(' ')
            # 顶点编号
            index = int(temp[0])
            SE[index] = torch.tensor([float(cc) for cc in temp[1:]])
        
    return SE


def load_TE_initial(data):
    time = pd.DatetimeIndex(data.index)
    # (seq_len,)->(seq_len,1) value: 0~6 [day-of-week]
    dayofweek = torch.tensor(time.weekday)
    dayofweek = dayofweek.unsqueeze(-1)
    # (seq_len,)->(seq_len,1) value: 0~287 [T]
    timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // 300
    timeofday = torch.tensor(timeofday)
    timeofday = timeofday.unsqueeze(-1)
    # time(seq_len,2)
    time = torch.cat((dayofweek,timeofday), dim=1)

    return time


def load_data(conf):
    """
        OUTPUT: data
                - trainX: (num_sample, num_his, num_vertex)
                - trainTE: (num_sample, num_his + num_pred, 2)
                - trainY: (num_sample, num_pred, num_vertex)
                - valX: (num_sample, num_his, num_vertex)
                - valTE: (num_sample, num_his + num_pred, 2)
                - valY: (num_sample, num_pred, num_vertex)
                - testX: (num_sample, num_his, num_vertex)
                - testTE: (num_sample, num_his + num_pred, 2)
                - testY: (num_sample, num_pred, num_vertex)
                - SE: (num_vertex, dims)
                - mean: float
                - std: float
    """
    data = {}
    # Get Traffic Data
    df = pd.read_hdf(conf['traffic_file'])
    # [seq_len, num_vertex]
    traffic = torch.from_numpy(df.values)

    # train/val/test Split
    num_step = df.shape[0]
    train_step = round(conf['train_radio'] * num_step)
    val_step = round(conf['val_radio'] * num_step)
    test_step = round(conf['test_radio'] * num_step)
    train = traffic[:train_step]
    val = traffic[train_step:train_step+val_step]
    test = traffic[-test_step:]

    # X,Y
    num_his = conf['num_his']
    num_pred= conf['num_pred']
    trainX, data['trainY'] = Seq2Instance(train, num_his, num_pred)
    valX, data['valY'] = Seq2Instance(val,num_his,num_pred)
    testX, data['testY'] = Seq2Instance(test,num_his,num_pred)

    # Normalization
    mean, std = torch.mean(trainX), torch.std(trainX)
    data['trainX'] = (trainX - mean) / std
    data['valX'] = (valX - mean) / std
    data['testX'] = (testX - mean) / std
    data['mean'] = mean
    data['std'] = std

    # Get SE
    data['SE'] = load_SE(conf['SE_file'])
    # Get TE initial
    time = load_TE_initial(df)

    # train/val/test TE Split
    train = time[:train_step]
    val = time[train_step:train_step+val_step]
    test = time[-test_step:]
    # [num_sample, num_his+num_pred, 2]
    trainTE_his, trainTE_pred = Seq2Instance(train, num_his, num_pred)
    data['trainTE'] = torch.cat((trainTE_his, trainTE_pred), dim=1)
    valTE_his, valTE_pred = Seq2Instance(val, num_his, num_pred)
    data['valTE'] = torch.cat((valTE_his, valTE_pred), dim=1)
    testTE_his, testTE_pred = Seq2Instance(test, num_his, num_pred)
    data['testTE'] = torch.cat((testTE_his, testTE_pred), dim=1)

    return data


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


def plot_train_val_loss(train_total_loss, val_total_loss):
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
    file_name = f"./img/plot/train_val_loss_{timestamp}.png"
    plt.savefig(file_name)
    
    # 清除图形
    plt.close()