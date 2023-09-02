from torchvision import datasets, transforms

def download_mnist_data():
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root = 'mnist_train', train= True, transform= transform, download=True)
    test_data = datasets.MNIST(root = 'mnist_test', train= False, transform= transform, download=True)

    print("Train Data Size: {}".format(len(train_data)))
    print("Test Data Size: {}".format(len(test_data)))

    return train_data, test_data
