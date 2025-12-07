from pureml.datasets.MNIST import MnistDataset
from pureml.models.neural_networks import MNIST_BEATER
from pureml.evaluation import accuracy


if __name__ == "__main__":
    train_set = MnistDataset(mode="train")
    test_set = MnistDataset(mode="test")
    model = MNIST_BEATER()
    model.fit(train_set, batch_size=16, num_epochs=50)

    acc = accuracy(model.eval(), test_set, batch_size=16)

    print(f"Test accuracy: {acc * 100}")
    # ------------ OUTCOME ------------
    """
            Test accuracy: 98.02
    """
    # ---------------------------------
