import pandas as pd
import numpy as np
# Get Traffic Data
df = pd.read_hdf("data/PeMS-BAY/PeMSbay.h5")
# [seq_len, num_vertex]
traffic = df.values

print(traffic.shape)
np.savez("data/PeMS-BAY/PeMS-BAY.npz", data=traffic)
# time = pd.DatetimeIndex(df.index)
# # (seq_len,)->(seq_len,1) value: 0~6 [day-of-week]
# dayofweek = time.weekday
# dayofweek = np.array(dayofweek)
# print(type(dayofweek))
# print(dayofweek.shape)
# dayofweek = np.expand_dims(dayofweek, axis=-1)
# print(dayofweek[:289])
# timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // 300
# timeofday = np.array(timeofday)
# timeofday = np.expand_dims(timeofday, axis=-1)
# print(timeofday[:289])
# time = np.concatenate((dayofweek, timeofday), axis=-1)
# print(time.shape)
# # print(time[:289])
# np.savez("data/PeMS-BAY/TE_PeMS-BAY.npz", data=time)
# dayofweek = dayofweek.unsqueeze(-1)
# # (seq_len,)->(seq_len,1) value: 0~287 [T]
# timeofday = (time.hour * 3600 + time.minute * 60 + time.second) // 300
# timeofday = torch.tensor(timeofday)
# timeofday = timeofday.unsqueeze(-1)
# # time(seq_len,2)
# time = torch.cat((dayofweek,timeofday), dim=1)