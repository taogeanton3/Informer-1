import pandas as pd
from torch.utils.data import Dataset

from utils.timefeature import timefeature


class MyDataset(Dataset):
    def __init__(self, df, scaler, seq_len=96, label_len=48, pred_len=24):
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.scaler = scaler

        self.__read_data__(df=df)

    def __read_data__(self, df):
        data = df.iloc[:, 1:].values.reshape(len(df), -1)
        data = self.scaler.transform(data)
        df["date"] = pd.to_datetime(df.iloc[:, 0])
        stamp = timefeature(df)

        self.data_x = data
        self.data_y = data
        self.stamp = stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.stamp[s_begin:s_end]
        seq_y_mark = self.stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
