import torch



def run_model(model, test_data, device: str):
    # model 실행 및 결괄르 얻기 위함이므로 no_grad
    # no_grad: 역전파 X
    with torch.no_grad():
        X_test = test_data.test_data.view(len(test_data), 1, 28, 28).float().to(device)
        Y_test = test_data.test_labels.to(device)

        prediction = model(X_test)
        
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())