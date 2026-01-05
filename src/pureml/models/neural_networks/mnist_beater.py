"""Reference MNIST MLP model (Affine-ReLU-Affine) with training loop.

Defines a tiny network (784→256→10) using Affine+ReLU blocks, SGD optimizer,
and CCE loss. `predict` returns logits in train mode and class indices in eval;
`fit` trains over a Dataset via DataLoader with shuffling and zeroing grads per step."""
import logging

from ...base import NN
from ...machinery import Tensor
from ...activations import relu
from ...layers import Affine
from ...training_utils import Dataset, DataLoader
from ...losses import CCE
from ...optimizers import SGD


_logger = logging.getLogger(__name__)


class MNIST_BEATER(NN):

    def __init__(self) -> None:
        self.L1 = Affine(28*28, 256)
        self.L2 = Affine(256, 10)
        self.train()

    def predict(self, x: Tensor) -> Tensor:
        x = x.flatten(sample_ndim=2) # passing 2 because imgs in MNIST are 2D
        x = relu(self.L1(x))
        x = self.L2(x)
        if self._training:
            return x
        return x.argmax(axis=x.ndim-1) # argmax over the feature dim

    def fit(self, dataset: Dataset, batch_size: int, num_epochs: int) -> None:
        if not self._training:
            raise RuntimeError(
                "To fit the dataset, switch the model to training mode via model.train()."
            )

        optim = SGD(self.parameters, lr=0.01)
        for e in range(num_epochs):
            total_loss = 0.0
            total_examples = 0
            for X, Y in DataLoader(dataset, batch_size, shuffle=True):
                Y_hat = self(X)
                L = CCE(Y, Y_hat, from_logits=True)

                bs = X.shape[0]
                total_loss += float(L.data) * bs
                total_examples += bs

                L.backward()
                optim.step()
                optim.zero_grad()
            epoch_loss = total_loss / max(total_examples, 1)
            _logger.info(f"Loss at epoch {e+1}: {epoch_loss:.6f}")
