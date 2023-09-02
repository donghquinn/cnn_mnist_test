import torch

def criterion_and_optimizer(model, device: str, learning_rate: float):
    # 비용 함수소프트맥스 함수 포함되어져 있음.
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return criterion, optimizer