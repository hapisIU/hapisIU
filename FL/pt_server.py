import argparse
import flwr as fl
import torch
from pt_client import get_data, PTMLPClient


def get_eval_fn(model):
    # This `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        loss, _, accuracy_dict = model.evaluate(parameters)
        return loss, accuracy_dict

    return evaluate

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rounds", type=int, default=3, \
                        help="number of rounds to train")
    args = parser.parse_args()
    torch.random.manual_seed(42)
    model = PTMLPClient(split="val")
    strategy = fl.server.strategy.FedAvg( \
        eval_fn=get_eval_fn(model), \
        )
    fl.server.start_server("[::]:8080", strategy=strategy, \
                           config={"num_rounds": args.rounds})
