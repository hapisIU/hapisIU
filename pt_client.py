import argparse
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.metrics
import torch
import torch.nn as nn
from functools import reduce
import flwr as fl
class PTMLPClient(fl.client.NumPyClient, nn.Module):
    def __init__(self, dim_in=4, dim_h=32, \
                 num_classes=3, lr=3e-4, split="alice"):
        super(PTMLPClient, self).__init__()
        self.dim_in = dim_in
        self.dim_h = dim_h
        self.num_classes = num_classes
        self.split = split

        self.w_xh = nn.Parameter(torch.tensor( \
            torch.randn(self.dim_in, self.dim_h) \
            / np.sqrt(self.dim_in * self.dim_h)) \
            )
        self.w_hh = nn.Parameter(torch.tensor( \
            torch.randn(self.dim_h, self.dim_h) \
            / np.sqrt(self.dim_h * self.dim_h)) \
            )
        self.w_hy = nn.Parameter(torch.tensor( \
            torch.randn(self.dim_h, self.num_classes) \
            / np.sqrt(self.dim_h * self.num_classes)) \
            )
        self.lr = lr

    def get_parameters(self):
        my_parameters = np.append( \
            self.w_xh.reshape(-1).detach().numpy(), \
            self.w_hh.reshape(-1).detach().numpy() \
            )
        my_parameters = np.append( \
            my_parameters, \
            self.w_hy.reshape(-1).detach().numpy() \
            )
        return my_parameters

    def set_parameters(self, parameters):
        parameters = np.array(parameters)
        total_params = reduce(lambda a, b: a * b, \
                              np.array(parameters).shape)
        expected_params = self.dim_in * self.dim_h \
                          + self.dim_h ** 2 \
                          + self.dim_h * self.num_classes

        start = 0
        stop = self.dim_in * self.dim_h
        self.w_xh = nn.Parameter(torch.tensor( \
            parameters[start:stop]) \
                                 .reshape(self.dim_in, self.dim_h).float() \
                                 )

        start = stop
        stop += self.dim_h ** 2
        self.w_hh = nn.Parameter(torch.tensor( \
            parameters[start:stop]) \
                                 .reshape(self.dim_h, self.dim_h).float() \
                                 )
        start = stop
        stop += self.dim_h * self.num_classes
        self.w_hy = nn.Parameter(torch.tensor( \
            parameters[start:stop]) \
                                 .reshape(self.dim_h, self.num_classes).float() \
                                 )
        self.act = torch.relu

        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.act(torch.matmul(x, self.w_xh))
        x = self.act(torch.matmul(x, self.w_hh))
        x = torch.matmul(x, self.w_hy)
        return x

    def get_loss(self, x, y):
        prediction = self.forward(x)
        loss = self.loss_fn(prediction, y)
        return loss

    def fit(self, parameters, config=None, epochs=10):
        self.set_parameters(parameters)
        x, y = get_data(split=self.split)
        x, y = torch.tensor(x).float(), torch.tensor(y).long()
        self.train()
        for ii in range(epochs):
            self.optimizer.zero_grad()
            loss = self.get_loss(x, y)
            loss.backward()
            self.optimizer.step()

        loss, _, accuracy_dict = self.evaluate(self.get_parameters())
        return self.get_parameters(), len(y), \
               {"loss": loss, "accuracy": \
                   accuracy_dict["accuracy"]}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        val_x, val_y = get_data(split="val")
        val_x = torch.tensor(val_x).float()
        val_y = torch.tensor(val_y).long()

        self.eval()
        prediction = self.forward(val_x)

        loss = self.loss_fn(prediction, val_y).detach().numpy()

        prediction_class = np.argmax( \
            prediction.detach().numpy(), axis=-1)

        accuracy = sklearn.metrics.accuracy_score( \
            val_y.numpy(), prediction_class)

        return float(loss), len(val_y), \
               {"accuracy": float(accuracy)}


def get_data(split="all"):
    x, y = sklearn.datasets.load_iris(return_X_y=True)

    np.random.seed(42);
    np.random.shuffle(x)
    np.random.seed(42);
    np.random.shuffle(y)
    val_split = int(0.2 * x.shape[0])
    train_split = (x.shape[0] - val_split) // 2
    eval_x, eval_y = x[:val_split], y[:val_split]

    alice_x, alice_y = x[val_split:val_split + train_split], y[val_split:val_split + train_split]

    bob_x, bob_y = x[val_split + train_split:], y[val_split + train_split:]

    train_x, train_y = x[val_split:], y[val_split:]

    if split == "all":
        return train_x, train_y
    elif split == "alice":
        return alice_x, alice_y
    elif split == "bob":
        return bob_x, bob_y
    elif split == "val":
        return eval_x, eval_y
    else:
        print("error: split not recognized.")
        return None


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split", type=str, default="alice", \
                        help="The training split to use, options are 'alice', 'bob', or 'all'")

    args = parser.parse_args()
    torch.random.manual_seed(42)

    fl.client.start_numpy_client("localhost:8080", client=PTMLPClient(split=args.split))