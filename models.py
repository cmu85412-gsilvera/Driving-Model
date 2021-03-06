from typing import Any, Dict, List, Optional, Tuple
import os
import torch
import numpy as np
import time
from visualizer import plot_vector_vs_time, plot_overlaid, plot_versus
from model_utils import (
    visualize_importance,
    compute_importances,
    seed_everything,
    results_dir,
)


def print_line():
    print(40 * "*")


class DrivingModel(torch.nn.Module):
    def __init__(self, features: Dict[str, List[str]]):
        super().__init__()
        self.steering_model = SteeringModel(features["steering"])
        self.throttle_model = ThrottleModel(features["throttle"])
        self.brake_model = BrakeModel(features["brake"])

    def load_from_cache(self):
        print("Loading driving model...")
        # load steering model
        steering_ckpt: str = os.path.join(results_dir, "steering.model.pt")
        assert os.path.exists(steering_ckpt)
        self.steering_model.load_from_ckpt(torch.load(steering_ckpt))
        # load throttle model
        throttle_ckpt: str = os.path.join(results_dir, "throttle.model.pt")
        assert os.path.exists(throttle_ckpt)
        self.throttle_model.load_from_ckpt(torch.load(throttle_ckpt))
        # load brake model
        brake_ckpt: str = os.path.join(results_dir, "brake.model.pt")
        assert os.path.exists(brake_ckpt)
        self.brake_model.load_from_ckpt(torch.load(brake_ckpt))
        print("...Driving model loading complete")

    def forward(
        self, x_steering, x_throttle, x_brake
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steer = np.squeeze(
            self.steering_model(torch.Tensor(x_steering)).detach().numpy()
        )
        throttle = np.squeeze(
            self.throttle_model(torch.Tensor(x_throttle)).detach().numpy()
        )
        brake = np.squeeze(self.brake_model(torch.Tensor(x_brake)).detach().numpy())

        # computes some symbolic logic by ensuring certain characteristics of the model

        # normalize outputs
        steer = (steer - np.mean(steer)) / np.std(steer)
        throttle = (throttle - np.mean(throttle)) / np.std(throttle)
        brake = (brake - np.mean(brake)) / np.std(brake)

        throttle[throttle < 0] = 0  # no negative throttle
        brake[brake < 0] = 0  # no negative brake

        # ensure throttle and brake don't occur simultaneously
        brake_and_throttle = (brake > 0) & (throttle > 0)
        more_brake = brake > throttle
        more_throttle = brake < throttle
        brake[brake_and_throttle] *= more_brake[brake_and_throttle]
        throttle[brake_and_throttle] *= more_throttle[brake_and_throttle]

        return (steer, throttle, brake)

    def train(self):
        self.steering_model.train()
        self.throttle_model.train()
        self.brake_model.train()

    def eval(self):
        self.steering_model.eval()
        self.throttle_model.eval()
        self.brake_model.eval()

    def begin_training(
        self,
        X: Dict[str, np.ndarray],
        Y: Dict[str, np.ndarray],
        Xt: Dict[str, np.ndarray],
        Yt: Dict[str, np.ndarray],
        t: np.ndarray,
    ) -> None:
        self.train()
        self.steering_model.train_model(
            X["steering"], Y["steering"], Xt["steering"], Yt["steering"], t
        )
        self.throttle_model.train_model(
            X["throttle"], Y["throttle"], Xt["throttle"], Yt["throttle"], t
        )
        self.brake_model.train_model(
            X["brake"], Y["brake"], Xt["brake"], Yt["brake"], t
        )

    def begin_evaluation(
        self, X: Dict[str, np.ndarray], Y: Dict[str, np.ndarray], t: np.ndarray,
    ) -> None:
        self.eval()
        self.steering_model.test_model(X["steering"], Y["steering"], t)
        self.throttle_model.test_model(X["throttle"], Y["throttle"], t)
        self.brake_model.test_model(X["brake"], Y["brake"], t)

    def output(
        self,
        training_data: Dict[str, Any],
        test_data: Dict[str, Any],
        t_train: np.ndarray,
        t_test: np.ndarray,
    ):
        t = np.concatenate((t_train, t_test))

        # TODO: compute overall model
        y_pred_train = self.forward(
            training_data["X"]["steering"],
            training_data["X"]["throttle"],
            training_data["X"]["brake"],
        )
        y_pred_test = self.forward(
            test_data["X"]["steering"],
            test_data["X"]["throttle"],
            test_data["X"]["brake"],
        )

        def norm(x):
            return (x - np.mean(x)) / np.std(x)

        actual_data = {}
        for i, k in enumerate(["steering", "throttle", "brake"]):
            Y_train = norm(training_data["Y"][k])
            Y_test = norm(test_data["Y"][k])
            if k != "steering":
                Y_train -= Y_train.min()
                Y_test -= Y_test.min()
            actual_data[k] = Y_test
            plot_overlaid(
                data=[Y_train, y_pred_train[i]],
                t=t_train,
                title=f"combined_training_{k}",
                subtitles=["pred", "actual"],
            )
            plot_overlaid(
                data=[Y_test, y_pred_test[i]],
                t=t_test,
                title=f"combined_testing_{k}",
                subtitles=["pred", "actual"],
                colours=["k", "g"],
            )

        plot_vector_vs_time(
            xyz=np.array(y_pred_test).T,
            t=t_test,
            ax_titles=["steer", "throttle", "brake"],
            title="Driving model predictions",
        )
        plot_vector_vs_time(
            xyz=np.array(
                [
                    actual_data["steering"],
                    actual_data["throttle"],
                    actual_data["brake"],
                ]
            ).T,
            t=t_test,
            ax_titles=["steer", "throttle", "brake"],
            title="Driving model ground truth",
            col="g",
        )


class SymbolModel(torch.nn.Module):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.loss_fn = torch.nn.MSELoss()
        self.num_epochs: int = 50
        self.lr = 0.001
        self.optimizer_type = torch.optim.Adam
        self.importances = None

    def init_optim(self):
        self.optimizer = self.optimizer_type(self.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer)

    def train_model(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Xt: np.ndarray,
        Yt: np.ndarray,
        t: np.ndarray,
    ) -> None:
        print_line()
        seed_everything()
        print(f"Starting {self.name} model training for {self.num_epochs} epochs...")
        acc_thresh = np.mean(np.abs(Yt))
        accs = []
        losses = []
        for epoch in range(self.num_epochs):
            start_t = time.time()
            """train model"""
            self.train()
            train_loss = 0
            for ix, x in enumerate(X):
                self.optimizer.zero_grad()
                data = torch.Tensor(x)
                desired = torch.Tensor([Y[ix]])
                outputs = self.forward(data)
                loss = self.loss_fn(outputs, desired)
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            """test model"""
            test_loss = 0
            correct = 0
            with torch.no_grad():
                self.eval()
                for ix, x in enumerate(Xt):
                    data = torch.Tensor(x)
                    desired = torch.Tensor([Yt[ix]])
                    outputs = self.forward(data)
                    correct += 1 if torch.abs(outputs - desired) < acc_thresh else 0
                    loss_crit = self.loss_fn(outputs, desired)
                    test_loss += loss_crit.item()
                acc = 100 * correct / len(Yt)
                accs.append(acc)
                losses.append(test_loss)
            self.scheduler.step(test_loss)
            print(
                f"Epoch {epoch} \t Train: {train_loss:4.3f} \t Test: {test_loss:4.3f}"
                f"\t Acc: {acc:2.1f}% in {time.time() - start_t:.2f}s"
            )
            full_predictions = np.array(
                [np.squeeze(self.forward(torch.Tensor(X)).detach().numpy()), Y]
            ).T
            plot_vector_vs_time(
                xyz=full_predictions,
                t=t,
                title=f"{self.name}.train.{epoch}",
                ax_titles=["pred", "actual"],
                silent=True,
            )
        importances = self.visualize_importances(Xt)  # X or Xt
        self.plot_accs_losses(accs, losses)
        filename: str = os.path.join(results_dir, f"{self.name}.model.pt")
        print(f"saving state dict to {filename}")
        torch.save(
            {
                "state_dict": self.state_dict(),
                "accs": accs,
                "losses": losses,
                "importances": importances,
            },
            filename,
        )

    def plot_accs_losses(self, accs: List[float], losses: List[float]) -> None:
        plot_versus(
            data_x=np.arange(len(losses)),
            data_y=losses,
            name_x="epochs",
            name_y=f"loss ({self.name})",
            lines=True,
        )
        plot_versus(
            data_x=np.arange(len(losses)),
            data_y=accs,
            name_x="epochs",
            name_y=f"accuracy ({self.name})",
            lines=True,
        )

    def load_from_ckpt(self, data: Dict[str, Any]) -> None:
        self.load_state_dict(data["state_dict"])
        accs = data["accs"]
        losses = data["losses"]
        self.importances = data["importances"]
        self.plot_accs_losses(accs, losses)
        print(f"Loaded {self.name} model!")

    def test_model(
        self, X: np.ndarray, Y: np.ndarray, t: np.ndarray,
    ):
        print_line()
        print(f"Beginning {self.name} test")
        y_pred = np.squeeze(self.forward(torch.Tensor(X)).detach().numpy())
        assert y_pred.shape == Y.shape
        pred_vs_actual = np.array([y_pred, Y]).T
        plot_vector_vs_time(
            xyz=pred_vs_actual,
            t=t,
            title=f"{self.name}.test",
            ax_titles=["pred", "actual"],
        )
        self.visualize_importances(X)

    def visualize_importances(self, X):
        assert hasattr(self, "feature_names")
        feature_names_small = [f[f.find("_") + 1 :] for f in self.feature_names]
        if self.importances is None:
            self.importances = compute_importances(self, torch.Tensor(X))
        visualize_importance(
            feature_names_small, self.importances, title=f"importances.{self.name}"
        )
        return self.importances

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "network")
        return self.network(x)


class SteeringModel(SymbolModel):
    def __init__(self, features: List[str]):
        super().__init__("steering")
        self.feature_names = features
        self.in_dim = len(features)
        self.out_dim = 1  # outputting only a single scalar
        layers = [
            torch.nn.Linear(self.in_dim, 64),
            torch.nn.Linear(64, 128),
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, self.out_dim),
        ]
        self.network = torch.nn.Sequential(*layers)
        self.init_optim()  # need to initalize optimizer after creating the network


class ThrottleModel(SymbolModel):
    def __init__(self, features: List[str]):
        super().__init__("throttle")
        self.feature_names = features
        self.in_dim = len(features)
        self.loss_fn = torch.nn.L1Loss()  # more resistant to outliers
        self.out_dim = 1  # outputting only a single scalar
        layers = [
            torch.nn.Linear(self.in_dim, 128),
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),  # only positive
            torch.nn.Linear(256, self.out_dim),
        ]
        self.lr = 0.001
        self.optimizer_type = torch.optim.Adagrad
        self.network = torch.nn.Sequential(*layers)
        self.init_optim()  # need to initalize optimizer after creating the network


class BrakeModel(SymbolModel):
    def __init__(self, features: List[str]):
        super().__init__("brake")
        self.feature_names = features
        self.in_dim = len(features)
        self.out_dim = 1  # outputting only a single scalar
        self.loss_fn = torch.nn.L1Loss()  # more resistant to outliers
        layers = [
            torch.nn.Linear(self.in_dim, 128),
            torch.nn.Linear(128, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),  # only positive
            torch.nn.Linear(256, self.out_dim),
        ]
        self.lr = 0.01
        self.optimizer_type = torch.optim.Adagrad
        self.network = torch.nn.Sequential(*layers)
        self.init_optim()  # need to initalize optimizer after creating the network
