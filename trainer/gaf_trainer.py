import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.trafficdataset import TrafficDataset
from utils.utils import load_config, Seq2Instance, count_parameters
from utils.utils import plot_train_val_loss, metrics
from trainer.base_trainer import BaseTrainer
from model.gaf import GAF
import time
from datetime import datetime
import ipdb
import os



class GAFTrainer(BaseTrainer):
    def __init__(self, cfg_file):
        self.conf = load_config(cfg_file)
        self.device = self.load_device()
        self.SE = self.load_SE()
        

    def load_SE(self):
        SE_file = f"data/{self.conf['dataset_name']}/SE_{self.conf['dataset_name']}.txt"
        with open(SE_file, mode='r') as f:
            lines = f.readlines()
            # V=325,D=64
            num_vertex, dims = map(int, lines[0].split(' '))
            SE = torch.zeros((num_vertex, dims), dtype=torch.float32)
            for line in lines[1:]:
                parts = line.split(' ')
                # 顶点编号
                index = int(parts[0])
                values = list(map(float, parts[1:]))
                SE[index] = torch.tensor(values, dtype=torch.float32)
        
        return SE.to(self.device)


    def load_data(self):
        """
        data
            - trainX: (num_sample, num_his, num_vertex)
            - trainTE: (num_sample, num_his + num_pred, 2)
            - trainY: (num_sample, num_pred, num_vertex)
            - valX: (num_sample, num_his, num_vertex)
            - valTE: (num_sample, num_his + num_pred, 2)
            - valY: (num_sample, num_pred, num_vertex)
            - testX: (num_sample, num_his, num_vertex)
            - testTE: (num_sample, num_his + num_pred, 2)
            - testY: (num_sample, num_pred, num_vertex)
            - mean: float
            - std: float
        """
        data = {}

        # Get Traffic Data
        TE_file = f"data/{self.conf['dataset_name']}/TE_{self.conf['dataset_name']}.npz"
        traffic_file = f"data/{self.conf['dataset_name']}/{self.conf['dataset_name']}.npz"
        
        # [seq_len, num_vertex]
        traffic = np.load(traffic_file)['data']
        traffic_data = torch.from_numpy(traffic)
        
        # train/val/test Split
        seq_len = traffic.shape[0]
        train_step = round(self.conf['train_radio'] * seq_len)
        val_step = round(self.conf['val_radio'] * seq_len)
        test_step = round(self.conf['test_radio'] * seq_len)
        train = traffic_data[:train_step]
        val = traffic_data[train_step:train_step+val_step]
        test = traffic_data[-test_step:]

        # X,Y
        num_his = self.conf['num_his']
        num_pred= self.conf['num_pred']
        trainX, data['trainY'] = Seq2Instance(train, num_his, num_pred)
        valX, data['valY'] = Seq2Instance(val, num_his, num_pred)
        testX, data['testY'] = Seq2Instance(test, num_his, num_pred)

        # Normalization
        mean, std = torch.mean(trainX), torch.std(trainX)
        data['trainX'] = (trainX - mean) / std
        data['valX'] = (valX - mean) / std
        data['testX'] = (testX - mean) / std

        self.mean = mean.clone().detach().to(self.device)
        self.std = std.clone().detach().to(self.device)

        # Get TE initial
        time = np.load(TE_file)['data']
        time = torch.from_numpy(time)

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

        # 加载数据集
        train_dataset = TrafficDataset(data['trainX'], data['trainY'], data['trainTE'])
        val_dataset = TrafficDataset(data['valX'], data['valY'], data['valTE'])
        test_dataset = TrafficDataset(data['testX'], data['testY'], data['testTE'])
        # dataloader
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.conf['batch_size'],
                                       shuffle=True,
                                       num_workers=self.conf['num_workers'],
                                       pin_memory=True)
        self.val_loader = DataLoader(val_dataset,
                                       batch_size=self.conf['batch_size'],
                                       shuffle=False,
                                       num_workers=self.conf['num_workers'],
                                       pin_memory=True)
        self.test_loader = DataLoader(test_dataset,
                                       batch_size=self.conf['batch_size'],
                                       shuffle=False,
                                       num_workers=self.conf['num_workers'],
                                       pin_memory=True)
        
        
    def train_epoch(self, epoch):
        total_loss = 0
        self.model.train()
        t_begin = time.time()
        
        for batch_index, (x, y, te) in enumerate(self.train_loader):
            # ipdb.set_trace()
            X = x.to(self.device)
            Y = y.to(self.device)
            TE = te.to(self.device)
            Y_hat = self.model(X, TE)
            Y_hat = Y_hat * self.std + self.mean

            loss_batch = self.loss_criterion(Y_hat, Y)

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()
            
            total_loss += loss_batch.item()
            # print loss
            if (batch_index + 1) % 10 == 0:
                print(f"[Training] Epoch:{epoch:<5}, Batch:{batch_index+1:<5}, MAE Loss:{loss_batch.item():.4f}")
        t_end = time.time() - t_begin

        return total_loss / len(self.train_loader), t_end
    

    def validate_epoch(self, epoch):
        total_loss = 0
        self.model.eval()
        t_begin = time.time()
        
        with torch.no_grad():
            for batch_index, (x, y, te) in enumerate(self.val_loader):
                X = x.to(self.device)
                Y = y.to(self.device)
                TE = te.to(self.device)
                Y_hat = self.model(X, TE)
                Y_hat = Y_hat * self.std + self.mean
                loss_batch = self.loss_criterion(Y_hat, Y)
                total_loss += loss_batch.item()
                # print loss
                if (batch_index + 1) % 10 == 0:
                    print(f"[Valitate] Epoch:{epoch:<5}, Batch:{batch_index+1:<5}, MAE Loss:{loss_batch.item():.4f}")

        t_end = time.time() - t_begin

        val_loss = total_loss / len(self.val_loader)
        return (val_loss, t_end)
    
    
    def test_epoch(self, plot=False):
        self.model.eval()
        t_begin = time.time()
        total_mae = 0
        total_rmse = 0
        total_mape = 0
        if plot:
            y_hat_list = []
            y_list = []
            x_list = []

        with torch.no_grad():
            for batch_index, (x, y, te) in enumerate(self.test_loader):
                X = x.to(self.device)
                Y = y.to(self.device)
                TE = te.to(self.device)
                Y_hat = self.model(X, TE)
                Y_hat = Y_hat * self.std + self.mean

                y_hat = Y_hat.clone().detach().cpu()
                y = Y.clone().detach().cpu()
                if plot:
                    newx = X * self.std + self.mean
                    newx = newx.clone().detach().cpu()
                    x_list.append(newx.numpy())
                    y_list.append(y.numpy())
                    y_hat_list.append(y_hat.numpy())
                mae, rmse, mape = metrics(y_hat, y)
                total_mae += mae
                total_rmse += rmse
                total_mape += mape
                # print loss
                if (batch_index + 1) % 10 == 0:
                    print(f"[Valitate]Batch:{batch_index+1:<4}, MAE Loss:{mae:.4f}, RMSE Loss:{rmse:.4f}, MAPE Loss:{mape*100:.4f}")

        t_end = time.time() - t_begin

        if plot:
            y = np.concatenate(y_list, axis=0)
            y_hat = np.concatenate(y_hat_list, axis=0)
            x = np.concatenate(x_list, axis=0)
            np.savez(f"./data/y_hat/GAF_{self.conf['dataset_name']}_prediction.npz", y=y, y_hat=y_hat, x=x)


        n1 = len(self.test_loader)
        avg_mae = total_mae / n1
        avg_rmse = total_rmse / n1
        avg_mape = total_mape / n1
        return (avg_mae, avg_rmse, avg_mape, t_end)
        

    def train(self):
        self.load_SE()
        self.load_data()
        self.model = GAF(
                self.SE,
                self.conf
            ).to(self.device)
        self.setup_train()
        count_parameters(self.model)
        train_total_loss = []
        val_total_loss = []
        min_loss_val = float('inf')
        epochs = self.conf['max_epoch']
        T_begin = time.time()
        
        for epoch in range(1, epochs+1):
            # train
            epoch_train_loss, time_train_epoch = self.train_epoch(epoch)
            train_total_loss.append(epoch_train_loss)
            # valitate
            epoch_val_loss, time_val_epoch = self.validate_epoch(epoch)
            val_total_loss.append(epoch_val_loss)

            # train time log
            print("%s | Epoch: %03d/%03d, Training time: %.1f Seconds, Inference time:%.1f Seconds" % 
                (datetime.now().strftime('%Y-%m-%d_%H:%M:%S'), epoch, epochs, time_train_epoch, time_val_epoch))
            # train loss log
            print(f"Training loss: {epoch_train_loss:.4f}, Validation loss: {epoch_val_loss:.4f}")

            self.scheduler.step()
        
            # 更小的loss_val,保存模型参数
            if epoch_val_loss <= min_loss_val:
                min_loss_val = epoch_val_loss
                best_model = self.model.state_dict()
            # 最后一轮保存模型参数
            if epoch == epochs:
                final_model = self.model.state_dict()

        T_end = time.time() - T_begin

        # 保存最小val_loss的model
        self.model.load_state_dict(best_model)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        min_loss_val_str = "{:.2f}".format(min_loss_val).replace('.', 'point')
        model_file1 = f"./ckpt/GAF_{self.conf['dataset_name']}_epoch{epochs}_best_MinValLoss{min_loss_val_str}_{timestamp}.ckpt"
        torch.save(self.model.state_dict(), model_file1)

        # 保存最后一轮的model
        self.model.load_state_dict(final_model)
        epoch_val_loss_str = "{:.2f}".format(epoch_val_loss).replace('.', 'point')
        model_file2 = f"./ckpt/GAF_{self.conf['dataset_name']}_epoch{epochs}_final_ValLoss{epoch_val_loss_str}_{timestamp}.ckpt"
        torch.save(self.model.state_dict(), model_file2)

        print(f"Well Done! Total Cost {T_end/60:.2f} Minutes To Train!")
        print(f"model has been saved in: ")
        print(f"1. {model_file1}")
        print(f"2. {model_file2}")

        # 绘制loss图像
        plot_train_val_loss(train_total_loss, val_total_loss, self.conf['dataset_name'])


    def load_pretrained_model(self):
        ckpt_dir = './ckpt'
        ckpt_files = os.listdir(ckpt_dir)

        # Get selected model path
        model_path = None
        print("All pretrained model files as follows:")
        
        # 只保留当前数据集有关模型文件
        model_files=[]
        for file in ckpt_files:
            if self.conf['dataset_name'] == file.split('_')[1]:
                model_files.append(file)
        def extract_date(filename):
            return filename.split("_")[-2] + "-" + filename.split("_")[-1].split(".")[0]
        model_files.sort(key=extract_date, reverse=True)
        for i, file in enumerate(model_files,1):
            print(f"[{i}]--{file}")

        select_num = input("根据数字选择你推理过程中要使用的模型：")
        model_path = os.path.join(ckpt_dir, model_files[int(select_num)-1])
        print(f"选中的模型: {model_files[int(select_num)-1]}")
        
        # Load model
        self.load_SE()
        self.model = GAF(
                self.SE,
                self.conf
            ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f"model restored from {model_path}, start inference...\n")


    def delete_load_pretrained_model(self):

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
        self.model = torch.load(model_path, map_location=self.device)
        print(f"model restored from {model_path}, start inference...")


    def eval(self,p):
        self.load_data()
        self.load_pretrained_model()
        mae, rmse, mape, t_cost = self.test_epoch(plot=p)
        print(f"Inference Time: {(t_cost):.1f} Seconds")
        print("                MAE\t\tRMSE\t\tMAPE%")
        print("test             {:.3f}\t\t{:.3f}\t\t{:.3f}%".format(mae, rmse, mape * 100))




