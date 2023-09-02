def train_model(model, epoch: int, data_loader, device: str, optimizer, criterion, total_batch):
    for e in range(epoch):
        avg_cost = 0

        # 미니 배치 단위로 꺼내온다. X는 미니 배치, Y는 레이블.
        for X, Y in data_loader: 
            # image is already size of (28x28), no reshape
            # label is not one-hot encoded
            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            hypothesis = model(X)
            cost = criterion(hypothesis, Y)
            cost.backward()
            optimizer.step()

            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(e + 1, avg_cost))