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
            Y_hat.append(y_hat.clone())
            
        Y_hat = torch.cat(Y_hat, dim=0)
        Y_hat = Y_hat * std + mean

    res = Y_hat.cpu()
    
    return res


def load_pretrained_model(device):

    ckpt_dir = './ckpt'
    ckpt_files = os.listdir(ckpt_dir)

    # Get selected model path
    model_path = None
    print("All pretrained model files as follows:")
    for i, file in enumerate(ckpt_files, 1):
        print(f"{i}. {file}")
    while selected_model := int(input("Choose the model to load (input the corresponding number): ")):
        if 0 < selected_model <= len(ckpt_files):
            model_path = os.path.join(ckpt_dir, ckpt_files[selected_model-1])
            print(f"Selected model: {ckpt_files[selected_model-1]}")
            break
        else:
            print("Invalid selection, please enter the correct number.")
    
    # Load model
    model = torch.load(model_path, map_location=device)
    print(f"model restored from {model_path}, start inference...")

    return model


def inference(conf, data):
    # 0. Set device
    torch.cuda.set_device(conf['device_id'])
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    #    load data To GPU
    # data = {key: value.to(device) for key, value in data.items()}

    # 1. Load model
    model = load_pretrained_model(device)

    # 2. Inference
    # Y_hat_train = inference_one_epoch(model, conf, data, 1)
    # Y_hat_val = inference_one_epoch(model, conf, data, 2)
    t_begin = time.time()
    Y_hat_test = inference_one_epoch(model, conf, data, device, 3)
    t_end = time.time()

    # 3. Calculate metrics
    # mae_train, rmse_train, mape_train = metric(Y_hat_train, data['trainY'])
    # mae_val, rmse_val, mape_val = metric(Y_hat_val, data['valY'])
    mae_test, rmse_test, mape_test = metric(Y_hat_test, data['testY'])

    # 4. Print metrics
    print(f"Inference Time: {(t_end - t_begin):.1f}seconds")
    print("                MAE\t\tRMSE\t\tMAPE%")
    # print("train            {:.2f}\t\t{:.2f}\t\t{:.2f}%".format(mae_train, rmse_train, mape_train * 100))
    # print("val              {:.2f}\t\t{:.2f}\t\t{:.2f}%".format(mae_val, rmse_val, mape_val * 100))
    print("test             {:.2f}\t\t{:.2f}\t\t{:.2f}%".format(mae_test, rmse_test, mape_test * 100))


conf = load_config()
data = load_data(conf)
inference(conf, data)
