import torch
from data import download_mnist_data
from model import CnnModel
from operate import run_model
from train import train_model
from utils import criterion_and_optimizer

def main():
    epoch = 15
    batch_size = 100
    learning_rate = 0.001

    train_data, test_data = download_mnist_data()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Device: {}".format(device))

    model = CnnModel().to(device)

    criterion, optimizer = criterion_and_optimizer(model, device, learning_rate)

    data_loader = torch.utils.data.DataLoader(dataset=train_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)
    
    total_batch = len(data_loader)

    print("Total Batch: {}".format(total_batch))

    train_model(model, epoch, data_loader, device, optimizer, criterion, total_batch)

    run_model(model, test_data, device)

main()
