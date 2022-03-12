from model import WaveUnet
from dataset import MyDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

IN_CHANNEL_LIST = [1, 24, 28, 72, 96, 120, 144, 168, 192, 216, 240, 264]
OUT_CHANNEL_LIST = [24, 28, 72, 96, 120, 144, 168, 192, 216, 240, 264, 288]
DATASET_PATH = "D:\\voicebank"
EPOCH = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collate_fn(batch):
    input_list = []
    label_list = []
    for item in batch:
        input, label = item
        input_list.append(input)
        label_list.append(label)
    return input_list, label_list


if __name__ == "__main__":
    dataset = MyDataset(DATASET_PATH)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)
    net = WaveUnet(IN_CHANNEL_LIST, OUT_CHANNEL_LIST)
    loss_func = nn.MSELoss()
    best_loss = float('inf')
    net.to(DEVICE)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    for epoch in range(EPOCH):
        net.train()
        last_save = 0
        curr_num = 0
        for noisy, label in dataloader:
            noisy = noisy.to(DEVICE)
            pred = net(noisy)
            label = label.to(DEVICE)
            loss = loss_func(pred, label)

            print("epoch:{},\tloss:{},\tlast_save:{}\tnum:{}/{}".format(epoch,
                                                                        loss.item(),
                                                                        last_save,
                                                                        curr_num,
                                                                        len(dataloader)))
            if loss.item() < best_loss:
                torch.save(net.state_dict(), "best_model")
                # a = torch.load("best_model.pth")
                best_loss = loss.item()
                last_save = curr_num
                print("save model")

            curr_num += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
