import os
import sys
import logging
import numpy as np
from copy import deepcopy
from configparser import ConfigParser
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from factory import PredictFactory, UpdateFactory
from Evaluate.factory import EvaluateFactory

logging.getLogger().setLevel(logging.INFO)


class SingleTracker:
    def __init__(self, cfg: ConfigParser, init_x, init_P):
        self.t = 0
        self.cfg = cfg
        self.predictor = PredictFactory.get(cfg)
        self.updater = UpdateFactory.get(cfg)
        self.x = {0: init_x}
        self.P = {0: init_P}

    def predict(self):
        x = self.x[self.t]
        P = self.P[self.t]

        D = self.cfg.get("Predict", "D")
        D = D.split(",")
        D = list(map(float, D))

        J = self.predictor.get_J(x)
        Q = self.predictor.get_Q(x, D)

        x = self.predictor.F(x)
        P = np.dot(np.dot(J, P), J.T) + Q

        return x, P

    def update(self, x: np.array, P: np.array, meas=None):
        if meas is not None:
            D = self.cfg.get("Update", "D")
            D = D.split(",")
            D = list(map(float, D))

            hx = self.updater.H(x)
            G = self.updater.get_G(x)
            R = self.updater.get_R(meas, D)
            I = np.identity(n=x.shape[0])

            S = np.dot(np.dot(G, P), G.T) + R
            S = (S + S.T) / 2
            K = np.dot(np.dot(P, G.T), np.linalg.inv(S))

            delta = self.updater.get_delta(hx, meas)

            x = x + np.dot(K, delta)
            P = np.dot(I - np.dot(K, G), P)

        self.t += 1
        self.x[self.t] = deepcopy(x)
        self.P[self.t] = deepcopy(P)


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config/config.cfg")
    evaluater = EvaluateFactory.get(config)
    ground_truth = evaluater.create()

    meas = []
    R = np.identity(2) * float(config.get("Evaluate", "sigma"))

    init_x = np.array(
        [0, 0, np.pi / 2, -0.005, 6.28]
    )
    init_x[0:2] = ground_truth[0][0:2]

    # init_x = np.array(
    #     [0, 0, 0, 6.28]
    # )
    # init_x[0:2] = ground_truth[0][0:2]

    shape = init_x.shape[0]
    tracker = SingleTracker(config, init_x, np.ones((shape, shape)))

    pd = float(config.get("Evaluate", "pd"))

    for t, pos in ground_truth.items():
        z = np.zeros((3, ), dtype=float)
        z[0:2] = np.random.multivariate_normal(ground_truth[t][0:2], R)

        angle = ground_truth[t][2]

        meas_angle = np.random.normal(angle, 0.1)
        z[2] = meas_angle

        meas.append(z.copy())

        if t > 0:
            x, P = tracker.predict()
            if pd > np.random.rand():
                tracker.update(x=x, P=P, meas=z)
            else:
                tracker.update(x=x, P=P)

        logging.info(f"-----------time_{t}------------")
        logging.info(f"pos is {tracker.x[t][0:2]}, "
                     f"gt is {ground_truth[t][0:2]},"
                     f"meas is {meas[t][0:2]}...")
        logging.info(f"angle is {tracker.x[t][2]}, "
                     f"gt is {ground_truth[t][2]},"
                     f"meas is {meas[t][2]}......")
        logging.info(f"R is {1 / tracker.x[t][3]}")
        logging.info(f"velocity is {tracker.x[t][4]}")

    # plot
    end_time = len(meas)
    plt.figure()

    for t in range(end_time - 1):
        plt.plot([ground_truth[t][0], ground_truth[t + 1][0]],
                 [ground_truth[t][1], ground_truth[t + 1][1]], c="red")
        plt.plot([meas[t][0], meas[t + 1][0]],
                 [meas[t][1], meas[t + 1][1]], c="black")
        plt.plot([tracker.x[t][0], tracker.x[t + 1][0]],
                 [tracker.x[t][1], tracker.x[t + 1][1]], c="blue")

    plt.axis('equal')
    plt.xlabel(xlabel='x position/m')
    plt.ylabel(ylabel='y position/m')

    plt.figure()
    times = list(ground_truth.keys())
    plt.plot(times, [np.linalg.norm(tracker.x[t][0:2] - ground_truth[t][0:2]) for t in times], c="red")
    plt.plot(times, [np.linalg.norm(meas[t][0:2] - ground_truth[t][0:2]) for t in times], c="black")
    plt.show()

