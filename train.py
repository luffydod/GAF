import time
from datetime import datetime
import math
import torch


def train(model, conf, data, loss_criterion, optimizer, scheduler):
    """
        OUTPUT:
                - train_loss: list[], float,
                - validation_loss: list[], float.
    """

    # trainX: (num_sample, num_his, num_vertex)
    num_sample_train = data['trainX'].shape[0]
    num_sample_val = data['valX'].shape[0]
    batch_size_train = math.ceil(num_sample_train / conf['batch_size'])
    batch_size_val = math.ceil(num_sample_val / conf['batch_size'])

    train_total_loss = []
    val_total_loss = []
    min_loss_val = float('inf')

    T_begin = time.time()
    # Train & Valitation
    for epoch in range(1,conf['max_epoch']+1):
        # shuffle train data
        permutation = torch.randperm(num_sample_train)
        trainX = data['trainX'][permutation]
        trainTE = data['trainTE'][permutation]
        trainY = data['trainY'][permutation]

        # train
        loss_train = 0
        t_train_begin = time.time()
        model.train()
        for batch_index in range(batch_size_train):
            index_begin = batch_index * conf['batch_size']
            index_end = min(num_sample_train, (batch_index + 1) * conf['batch_size'])
            X = trainX[index_begin: index_end]
            TE = trainTE[index_begin: index_end]
            Y = trainY[index_begin: index_end]

            Y_hat = model(X, TE)
            Y_hat = Y_hat * data['std'] + data['mean']
            
            loss_batch = loss_criterion(Y_hat, Y)
            loss_train += float(loss_batch) * (index_end - index_begin)
            
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()
            # 清空 CUDA 缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # print loss
            if (batch_index + 1) % 5 == 0:
                print(f"[Training] Epoch:{epoch:<5}, Batch:{batch_index+1:<5}, Loss:{loss_batch:.4f}")
            del X, TE, Y, Y_hat, loss_batch
        
        loss_train /= num_sample_train
        train_total_loss.append(loss_train)
        t_train = time.time() - t_train_begin

        # val
        loss_val = 0
        t_val_begin = time.time()
        model.eval()
        with torch.no_grad():
            for batch_index in range(batch_size_val):
                index_begin = batch_index * conf['batch_size']
                index_end = min(num_sample_val, (batch_index + 1) * conf['batch_size'])
                X = data['valX'][index_begin: index_end]
                TE = data['valTE'][index_begin: index_end]
                Y = data['valY'][index_begin: index_end]
                Y_hat = model(X, TE)
                Y_hat = Y_hat * data['std'] + data['mean']
                
                loss_batch = loss_criterion(Y_hat, Y)
                loss_val += float(loss_batch) * (index_end - index_begin)
                del X, TE, Y, Y_hat, loss_batch

        loss_val /= num_sample_val
        val_total_loss.append(loss_val)
        t_val = time.time() - t_val_begin
        
        # train time log
        print("%s | Epoch: %04d/%d, Training time: %.1fseconds, Inference time:%.1fseconds" % 
              (datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), epoch, conf['max_epoch'], t_train, t_val))
        # train loss log
        print(f"Training loss: {loss_train:.4f}, Validation loss: {loss_val:.4f}")

        # 更小的loss_val,保存模型参数
        if loss_val <= min_loss_val:
            min_loss_val = loss_val
            best_model = model.state_dict()
        
        scheduler.step

    T_end = time.time()
    model.load_state_dict(best_model)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    min_loss_val_str = "{:.2f}".format(min_loss_val).replace('.', 'point')
    model_file = f"./ckpt/GGBond_epoch{conf['max_epoch']}_MinValLoss{min_loss_val_str}_{timestamp}.ckpt"
    torch.save(model, model_file)
    print(f"Well Done! Total Cost {(T_end-T_begin)/60:.2f} minutes To Train!")
    print(f"model has been saved in {model_file}.")

    return train_total_loss, val_total_loss