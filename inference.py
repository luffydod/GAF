import torch
import time
from datetime import datetime
import numpy as np
import math
import os
from utils.utils import metric, load_config, load_data

def inference_one_epoch(model, conf, data, num):
    # data
    std = data['std']
    mean = data['mean']
    x_str = ['trainX','valX','testX']
    te_str = ['trainTE','valTE','testTE']

    num_sample = data[x_str[num-1]].shape[0]
    batch_size_ = math.ceil(num_sample / conf['batch_size'])
    nX = data[x_str[num-1]]
    nTE = data[te_str[num-1]]

    with torch.no_grad():
        Y_hat = []
        for batch_index in range(batch_size_):
            start_index = batch_index * conf['batch_size']
            end_index = min(num_sample, (batch_index + 1) * conf['batch_size'])
            X = nX[start_index: end_index]
            TE = nTE[start_index: end_index]
            y_hat = model(X, TE)
            Y_hat.append(y_hat.detach().clone())
            del X, TE, y_hat
        Y_hat = torch.from_numpy(np.concatenate(Y_hat, axis=0))
        Y_hat = Y_hat * std + mean
    return Y_hat


def inference(conf, data):
    # trainX: (num_sample, num_his, num_vertex)

    # 定义ckpt目录路径
    ckpt_dir = './/ckpt'
    # 列出目录下的所有文件
    ckpt_files = os.listdir(ckpt_dir)
    # 打印所有模型参数文件
    print("模型参数文件列表：")
    for i, file in enumerate(ckpt_files, 1):
        print(f"{i}. {file}")
    while True:
        selected_model = input("请选择要加载的模型（输入相应数字）: ")
        # 根据用户选择加载模型
        try:
            selected_model_index = int(selected_model) - 1
            if 0 <= selected_model_index < len(ckpt_files):
                selected_model_path = os.path.join(ckpt_dir, ckpt_files[selected_model_index])
                print(f"您选择了加载模型：{ckpt_files[selected_model_index]}")
                break
            else:
                print("无效的选择，请输入正确的数字。")
        except ValueError:
            print("无效的选择，请输入数字。")
    # load model
    model = torch.load(selected_model_path, map_location=torch.device('cpu'))
    # map_loacation =torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print("model restored from %s" % selected_model_path)
    print("start inference...")

    Y_hat_train = inference_one_epoch(model, conf, data, 1)
    Y_hat_val = inference_one_epoch(model, conf, data, 2)
    t_begin = time.time()
    Y_hat_test = inference_one_epoch(model, conf, data, 3)
    t_end = time.time()

    print("testY.shape: ", data['testY'].shape)
    print("Y_hat_test.shape: ", Y_hat_test.shape)

    mae_train, rmse_train, mape_train = metric(Y_hat_train, data['trainY'])
    mae_val, rmse_val, mape_val = metric(Y_hat_val, data['valY'])
    mae_test, rmse_test, mape_test = metric(Y_hat_test, data['testY'])

    print("Inference Time: %.1fs" % (t_end - t_begin))
    print("                MAE\t\tRMSE\t\tMAPE")
    print("train            %.2f\t\t%.2f\t\t%.2f%%" % (mae_train, rmse_train, mape_train * 100))
    print("val              %.2f\t\t%.2f\t\t%.2f%%" % (mae_val, rmse_val, mape_val * 100))
    print("test             %.2f\t\t%.2f\t\t%.2f%%" % (mae_test, rmse_test, mape_test * 100))


conf = load_config()
data = load_data(conf)
inference(conf, data)
