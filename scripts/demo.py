import numpy as np
import pandas as pd
import torch
import h5py


with h5py.File("data/PEMS-bay-mini.h5",'r') as f:
    print(f.keys())
    speed = f['speed']
    print(type(speed))
    print(speed.keys())
    data1 = speed['block0_values']
    data2 = np.load("data/PEMS-bay.npy")
    print(f"data1.shape{data1.shape}, data2.shape{data2.shape}")
    if np.array_equal(data1, data2):
        print("True")
# Get Traffic Data
df = pd.read_hdf("data/PEMS-bay-mini.h5")
# # 截取最后 2880 行数据
# df_last = df.iloc[-2880:]

# # 将修改后的 DataFrame 保存为新的 HDF5 文件
# df_last.to_hdf("data/PEMS-bay-mini.h5", key='speed', mode='w')

print(df.head())  # 查看 DataFrame 的前几行
print(df.tail())  # 查看 DataFrame 的后几行
print(df.shape)  # 输出 DataFrame 的形状，即行数和列数
print(df.columns)  # 输出 DataFrame 的列名
print(df.dtypes)  # 输出 DataFrame 的每一列的数据类型

# [seq_len, num_vertex]
traffic = df.values
# print("traffic.shape = ", traffic.shape)
# data = numpy.load("data/PEMS-bay.npy")
# print(data.shape)

# # Get Traffic Data
# df = pd.read_hdf("data/PEMS-bay.h5")
# # [seq_len, num_vertex]
# traffic = df.values

with h5py.File("data/PEMS-bay.h5", 'r') as f:
    # f.keys() = [speed,]
    # f['speed']['axis0'] (325,)
    # f['speed']['axis1'] (52116,)
    # f['speed']['block0_items'] (325,)
    # f['speed']['block0_values'] (52116,325)
    data_axis0 = f['speed']['axis0'][:]
    data_axis1 = f['speed']['axis1'][-2880:]
    data_block0_items = f['speed']['block0_items'][:]
    data_block0_values = f['speed']['block0_values'][-2880:]

    # save new h5 files
    with h5py.File("data/PEMS-bay-mini.h5", 'w') as f_new:
        f_new.create_dataset('speed/axis0', data=data_axis0)
        f_new.create_dataset('speed/axis1', data=data_axis1)
        f_new.create_dataset('speed/block0_items', data=data_block0_items)
        f_new.create_dataset('speed/block0_values', data=data_block0_values)