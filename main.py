import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import MyDataset
from model.informer import Informer
from utils.setseed import set_seed

warnings.filterwarnings("ignore")

lr = 0.0001
epochs = 10
batch_size = 32
seq_len = 96
label_len = 48
pred_len = 24
rootpath = "./Informer/"
trainrate = 0.7

if __name__ == "__main__":
    writer = SummaryWriter(rootpath + "log/tensorboard/")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    df = pd.read_csv(rootpath + "data/ETT/ETTh1.csv")
    train = df.iloc[: int(trainrate * len(df)), :]
    test = df.iloc[int(trainrate * len(df)):, :]

    scaler = StandardScaler()
    scaler.fit(train.iloc[:, 1:].values)

    trainset = MyDataset(train, scaler, seq_len=96, label_len=48, pred_len=24)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    testset = MyDataset(test, scaler, seq_len=96, label_len=48, pred_len=24)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    model = Informer().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # show
    print("show...")
    for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(trainloader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
        dec_inp = (
            torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
        )
        with writer as w:
            w.add_graph(model, (batch_x, batch_x_mark, dec_inp, batch_y_mark))
        break

    # train
    print("train...")
    model.train()
    for e in range(epochs):
        losses = []
        for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(trainloader):
            optimizer.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            dec_inp = torch.zeros(
                [batch_y.shape[0], pred_len, batch_y.shape[-1]]
            ).float()
            dec_inp = (
                torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
            )

            pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            pred = pred[:, -pred_len:, :].to(device)
            true = batch_y[:, -pred_len:, :].to(device)

            loss = criterion(pred, true)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        print("Epochs:", e, " || train loss: %.4f" % np.mean(losses))

    torch.save(model, rootpath + "log/informer.pkl")

    # test
    print("test...")
    # torch.load("./Informer/log/informer.pkl").to(device)

    model.eval()
    losses = []
    trues, preds = [], []
    for (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(testloader):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float()
        batch_x_mark = batch_x_mark.float().to(device)
        batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = torch.zeros([batch_y.shape[0], pred_len, batch_y.shape[-1]]).float()
        dec_inp = (
            torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1).float().to(device)
        )

        pred = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

        preds.extend(pred.detach().cpu().numpy())
        trues.extend(batch_y.detach().cpu().numpy())

        pred = pred[:, -pred_len:, :].to(device)
        true = batch_y[:, -pred_len:, :].to(device)

        loss = criterion(pred, true)
        losses.append(loss.item())
    print("test loss: %.4f" % np.mean(losses))

    np.save(rootpath + "log/preds", np.array(preds))
    np.save(rootpath + "log/tures", np.array(trues))

    # show
    pred = np.load(rootpath + "log/preds.npy")
    true = np.load(rootpath + "log/tures.npy")

    print(pred.shape, true.shape)
    plt.plot(pred[0, -24:, -1], label="pred")
    plt.plot(true[0, -24:, -1], label="true")
    plt.legend()
    plt.savefig(rootpath + "img/show.png")
    plt.show()
