import numpy as np
from copy import deepcopy
from configparser import ConfigParser


class BasePredictor:
    def __init__(self, cfg: ConfigParser):
        self.t = float(cfg.get("Predict", "delta_t"))

    def F(self, x: np.array):
        raise NotImplementedError

    def get_J(self, x: np.array):
        raise NotImplementedError

    def get_Q(self, x: np.array, D):
        raise NotImplementedError


class CCVPredictor(BasePredictor):
    def F(self, x: np.array):
        return np.array(
            [
                x[0] + x[4] * np.cos(x[2]) * self.t,
                x[1] + x[4] * np.sin(x[2]) * self.t,
                x[2] - x[3] * x[4] * self.t,
                x[3],
                x[4]
            ], dtype=float
        )

    def get_J(self, x: np.array):
        return np.array(
            [
                [1, 0, -x[4] * np.sin(x[2]) * self.t, 0, np.cos(x[2]) * self.t],
                [0, 1, -x[4] * np.cos(x[2]) * self.t, 0, np.sin(x[2]) * self.t],
                [0, 0, 1, -x[4] * self.t, -x[3] * self.t],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ], dtype=float
        )

    def get_Q(self, x: np.array, D):
        Q = np.identity(5)
        t = self.t
        ak, av = D[0], D[1]

        Q[0, 0] = 0.25 * av * np.cos(x[2]) ** 2 * t ** 4
        Q[0, 1] = 0.25 * av * np.cos(x[2]) * np.sin(x[2]) * t ** 4
        Q[0, 2] = -0.25 * av * x[3] * np.cos(x[2]) * t ** 4
        Q[0, 3] = 0
        Q[0, 4] = 0.5 * av * np.cos(x[2]) * t ** 3

        Q[1, 1] = 0.25 * av * np.sin(x[2]) ** 2 * t ** 4
        Q[1, 2] = -0.25 * av * x[3] * np.sin(x[2]) * t ** 4
        Q[1, 3] = 0
        Q[1, 4] = 0.5 * av * np.sin(x[2]) * t ** 3

        Q[2, 2] = 0.25 * (x[4] ** 2 * ak + x[3] ** 2 * av) * t ** 4
        Q[2, 3] = -0.5 * ak * x[4] * t ** 3
        Q[2, 4] = -0.5 * av * x[3] * t ** 3

        Q[3, 3] = ak * t ** 2
        Q[3, 4] = 0

        Q[4, 4] = av * t ** 2

        for i in range(5):
            for j in range(i):
                Q[i, j] = Q[j, i]

        return Q


class CVPredictor(BasePredictor):
    def F(self, x: np.array):
        t = self.t

        F = np.array(
            [
                [1, 0, t, 0],
                [0, 1, 0, t],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=float
        )

        return np.dot(F, x)

    def get_J(self, x: np.array):
        t = self.t

        return np.array(
            [
                [1, 0, t, 0],
                [0, 1, 0, t],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=float
        )

    def get_Q(self, x, D):
        ax, ay = D[0], D[1]
        t = self.t
        Q = np.identity(4)

        Q[0, 0] = 0.25 * ax * t ** 4
        Q[0, 1] = 0
        Q[0, 2] = 0.5 * ax * t ** 3
        Q[0, 3] = 0

        Q[1, 1] = 0.25 * ay * t ** 3
        Q[1, 2] = 0
        Q[1, 3] = 0.5 * ay * t ** 3

        Q[2, 2] = ax * t ** 2
        Q[2, 3] = 0

        Q[3, 3] = ay * t ** 2

        return Q


class CCVInvPredictor(BasePredictor):
    def F(self, x_: np.array):
        x = deepcopy(x_)
        x[3] = 1 / x[3]

        return np.array(
            [
                x[0] + x[4] * np.cos(x[2]) * self.t,
                x[1] + x[4] * np.sin(x[2]) * self.t,
                x[2] - x[3] * x[4] * self.t,
                x[3],
                x[4]
            ], dtype=float
        )

    def get_J(self, x_: np.array):
        x = deepcopy(x_)
        x[3] = 1 / x[3]

        return np.array(
            [
                [1, 0, -x[4] * np.sin(x[2]) * self.t, 0, np.cos(x[2]) * self.t],
                [0, 1, -x[4] * np.cos(x[2]) * self.t, 0, np.sin(x[2]) * self.t],
                [0, 0, 1, -x[4] * self.t, -x[3] * self.t],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1]
            ], dtype=float
        )

    def get_Q(self, x_: np.array, D):
        x = deepcopy(x_)
        x[3] = 1 / x[3]

        Q = np.identity(5)
        t = self.t
        ak, av = D[0], D[1]

        Q[0, 0] = 0.25 * av * np.cos(x[2]) ** 2 * t ** 4
        Q[0, 1] = 0.25 * av * np.cos(x[2]) * np.sin(x[2]) * t ** 4
        Q[0, 2] = -0.25 * av * x[3] * np.cos(x[2]) * t ** 4
        Q[0, 3] = 0
        Q[0, 4] = 0.5 * av * np.cos(x[2]) * t ** 3

        Q[1, 1] = 0.25 * av * np.sin(x[2]) ** 2 * t ** 4
        Q[1, 2] = -0.25 * av * x[3] * np.sin(x[2]) * t ** 4
        Q[1, 3] = 0
        Q[1, 4] = 0.5 * av * np.sin(x[2]) * t ** 3

        Q[2, 2] = 0.25 * (x[4] ** 2 * ak + x[3] ** 2 * av) * t ** 4
        Q[2, 3] = -0.5 * ak * x[4] * t ** 3
        Q[2, 4] = -0.5 * av * x[3] * t ** 3

        Q[3, 3] = ak * t ** 2
        Q[3, 4] = 0

        Q[4, 4] = av * t ** 2

        for i in range(5):
            for j in range(i):
                Q[i, j] = Q[j, i]

        return Q
