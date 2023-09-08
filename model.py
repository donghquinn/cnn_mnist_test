import torch
from torch import nn

class CnnModel(nn.Module):
  def __init__(self):
    super(CnnModel, self).__init__()
    # 첫번째층
    # ImgIn shape=(?, 28, 28, 1)
    #    Conv     -> (?, 28, 28, 32)
    #    Pool     -> (?, 14, 14, 32)
    self.layer1 = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2))

    # 두번째층
    # ImgIn shape=(?, 14, 14, 32)
    #    Conv      ->(?, 14, 14, 64)
    #    Pool      ->(?, 7, 7, 64)
    self.layer2 = torch.nn.Sequential(
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2, stride=2))

    # 전결합층 7x7x64 inputs -> 10 outputs
    self.fc = torch.nn.Linear(7 * 7 * 64, 10, bias=True)

    # 전결합층 한정으로 가중치 초기화
    torch.nn.init.xavier_uniform_(self.fc.weight)

    # # 전결합층 한정으로 가중치 초기화
    # torch.nn.init.xavier_uniform_(self.fc.weight)

  def forward(self, x):
    out = self.layer1(x)
    out = self.layer2(out)

    # 전결합층을 위해서 Flatten
    out = out.view(out.size(0), -1)   

    # 전결합층 Fully Connected
    out = self.fc(out)
    
    return out