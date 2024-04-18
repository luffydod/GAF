import numpy as np
import datetime
# time.shape [seq_len, 2]
# data_path = "data/PeMS-BAY/PEMS-bay.npy"
dataset_name = "PeMS04"
data_path = f"data/{dataset_name}/PEMS04.npz"
data = np.load(data_path)['data']
print(data.shape)
newdata = data[:, :, 0]
newdata = np.squeeze(newdata)
print(newdata.shape)
np.savez(f"data/{dataset_name}/seq{dataset_name}.npz", data=newdata)
# seq_len = data.shape[0]
# # (16992, 307, 3)
# # 2018.01.01-2018.02.28
# # TE.shape [num_sample, num_his+num_pred, 2]
# start_date = datetime.datetime(2018,1,1)
# time_interval = datetime.timedelta(minutes=5)
# # 初始化日期特征序列
# day_of_week = np.zeros((seq_len, 1), dtype=np.int32)
# time_of_day = np.zeros((seq_len, 1), dtype=np.int32)
# # 生成日期特征序列
# for i in range(seq_len):
#     current_date = start_date + i * time_interval
#     day_of_week[i] = current_date.weekday() # 获取周几编号（0-6）
#     time_of_day[i] = (current_date.hour * 3600 + current_date.minute * 60 + current_date.second) // 300    # 获取当天时间编号（0-287）

# time = np.concatenate([day_of_week, time_of_day], axis=-1)
# print(time.shape)
# print(time[:288])

# np.savez("data/PeMS04/TE_PeMS04.npz", time)