from typing import Any, Dict, List, Optional, Tuple
import os
import torch
import numpy as np
import time
from visualizer import plot_vector_vs_time
from model_utils import visualize_importance, seed_everything, results_dir


def print_line():
    print(40 * "*")


class DrivingModel(torch.nn.Module):
    def __init__(self, features: Dict[str, List[str]]):
        super().__init__()
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print(
                f"Using NVidia GPU ({torch.cuda.get_device_name(0)}) for hardware acceleration"
            )
        self.steering_model = SteeringModel(features["steering"], self.device).to(
            self.device
        )
        self.throttle_model = ThrottleModel(features["throttle"], self.device).to(
            self.device
        )
        self.brake_model = BrakeModel(features["brake"], self.device).to(self.device)

    def load_from_cache(self):
        print("Loading driving model...")
        # load steering model
        steering_ckpt: str = os.path.join(results_dir, "steering.model.50.pt")
        assert os.path.exists(steering_ckpt)
        self.steering_model.load_state_dict(torch.load(steering_ckpt))
        # load throttle model
        throttle_ckpt: str = os.path.join(results_dir, "throttle.model.35.pt")
        assert os.path.exists(throttle_ckpt)
        self.throttle_model.load_state_dict(torch.load(throttle_ckpt))
        # load brake model
        brake_ckpt: str = os.path.join(results_dir, "brake.model.35.pt")
        assert os.path.exists(brake_ckpt)
        self.brake_model.load_state_dict(torch.load(brake_ckpt))
        print("...Driving model loading complete")

    def forward(
        self, x_steering, x_throttle, x_brake
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        steer = np.squeeze(
            self.steering_model(torch.Tensor(x_steering)).detach().cpu().numpy()
        )
        throttle = np.squeeze(
            self.throttle_model(torch.Tensor(x_throttle)).detach().cpu().numpy()
        )
        brake = np.squeeze(
            self.brake_model(torch.Tensor(x_brake)).detach().cpu().numpy()
        )

        # ensure no negative values
        throttle[throttle < 0] = 0
        brake[brake < 0] = 0

        # ensure throttle and brake don't occur simultaneously
        brake[throttle < brake] = 0
        throttle[brake < throttle] = 0

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
        # self.steering_model.train_model(
        #     X["steering"], Y["steering"], Xt["steering"], Yt["steering"], t
        # )
        self.throttle_model.train_model(
            X["throttle"], Y["throttle"], Xt["throttle"], Yt["throttle"], t
        )
        self.brake_model.train_model(
            X["brake"], Y["brake"], Xt["brake"], Yt["brake"], t
        )

    def begin_evaluation(
        self, X: Dict[str, np.ndarray], Y: Dict[str, np.ndarray], t: np.ndarray
    ) -> None:
        self.eval()
        self.steering_model.test_model(X["steering"], Y["steering"], t)
        self.throttle_model.test_model(X["throttle"], Y["throttle"], t)
        self.brake_model.test_model(X["brake"], Y["brake"], t)

    def symbolic_logic(
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


class SymbolModel(torch.nn.Module):
    def __init__(self, name: str, device: torch.DeviceObjType):
        super().__init__()
        self.name = name
        self.loss_fn = torch.nn.MSELoss()
        self.num_epochs: int = 35
        self.lr = 0.001
        self.optimizer_type = torch.optim.Adam
        self.device = device

    def init_optim(self):
        self.optimizer = self.optimizer_type(self.parameters(), lr=self.lr, device=self.device)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer).to(self.device)

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
        X_device = torch.Tensor(X).to(self.device)
        Y_device = torch.Tensor(Y).to(self.device)
        Xt_device = torch.Tensor(Xt).to(self.device)
        Yt_device = torch.Tensor(Yt).to(self.device)
        for epoch in range(self.num_epochs):
            start_t = time.time()
            """train model"""
            self.train()
            train_loss = 0
            for i in range(len(X)):
                self.optimizer.zero_grad()
                data = X_device[i]
                desired = Y_device[i]
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
                for i in range(len(Xt)):
                    data = Xt_device[i]
                    desired = Yt_device[i]
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
                [np.squeeze(self.forward(torch.Tensor(X)).detach().cpu().numpy()), Y]
            ).T
            plot_vector_vs_time(
                xyz=full_predictions,
                t=t,
                title=f"{self.name}.train.{epoch}",
                ax_titles=["pred", "actual"],
                silent=True,
            )
        filename: str = os.path.join(
            results_dir, f"{self.name}.model.{self.num_epochs}.pt"
        )
        print(f"saving state dict to {filename}")
        torch.save(self.state_dict(), filename)

    def test_model(
        self, X: np.ndarray, Y: np.ndarray, t: np.ndarray, visualize_importance=False
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

        if visualize_importance:
            self.visualize_importances(X)

    def visualize_importances(self, X):
        assert hasattr(self, "feature_names")
        feature_names_small = [f[f.find("_") + 1 :] for f in self.feature_names]
        visualize_importance(
            self, feature_names_small, torch.Tensor(X), title=f"{self.name} importances"
        )

    def forward(self, x: torch.Tensor):
        if x.device != self.device:
            return self.network(x.to(self.device))
        return self.network(x)


class SteeringModel(SymbolModel):
    def __init__(self, features: List[str], device: torch.DeviceObjType):
        super().__init__("steering", device)
        self.feature_names = features
        self.in_dim = len(features)
        self.out_dim = 1  # outputting only a single scalar
        self.num_epochs = 50
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
        self = self.to(device)


class ThrottleModel(SymbolModel):
    def __init__(self, features: List[str], device: torch.DeviceObjType):
        super().__init__("throttle", device)
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
        self.optimizer_type = torch.optim.Adagrad
        self.network = torch.nn.Sequential(*layers)
        self.init_optim()  # need to initalize optimizer after creating the network
        self = self.to(device)


class BrakeModel(SymbolModel):
    def __init__(self, features: List[str], device: torch.DeviceObjType):
        super().__init__("brake", device)
        self.feature_names = features
        self.in_dim = len(features)
        self.out_dim = 1  # outputting only a single scalar
        layers = [
            torch.nn.Linear(self.in_dim, 32),
            torch.nn.Linear(32, 64),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),  # only positive
            torch.nn.Linear(64, self.out_dim),
        ]
        self.lr = 0.01
        self.optimizer_type = torch.optim.Adagrad
        self.network = torch.nn.Sequential(*layers)
        self.init_optim()  # need to initalize optimizer after creating the network
        self = self.to(device)
