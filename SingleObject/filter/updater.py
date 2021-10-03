import numpy as np
from configparser import ConfigParser


class BaseUpdater:
    def __init__(self, cfg: ConfigParser):
        self.angle_pos = int(cfg.get("Update", "angle_pos"))

    def H(self, x: np.array):
        raise NotImplementedError

    def get_G(self, x: np.array):
        raise NotImplementedError

    def get_R(self, z, D):
        raise NotImplementedError

    def get_delta(self, x: np.array, z: np.array):
        delta = z - x

        if self.angle_pos >= 0:
            dot = np.arccos(np.cos(z[self.angle_pos] - x[self.angle_pos]))
            cross = np.sin(z[self.angle_pos] - x[self.angle_pos])
            if cross >= 0:
                delta[self.angle_pos] = dot
            else:
                delta[self.angle_pos] = -dot

        return delta


class LinearUpdater(BaseUpdater):
    def __init__(self, cfg):
        super(LinearUpdater, self).__init__(cfg)
        model = cfg.get("Update", "updater")
        model = model.split("_")
        col = int(model[1])
        raw = int(model[2])

        H1 = np.identity(raw)
        H2 = np.zeros((raw, col - raw))
        self._H = np.concatenate((H1, H2), axis=1)

    def H(self, x: np.array):
        return np.dot(self._H, x)

    def get_G(self, x: np.array):
        return self._H

    def get_R(self, z, D):
        raw = self._H.shape[0]
        assert len(D) == raw, "Wrong dimension!"

        R = np.identity(raw)
        for i in range(raw):
            R[i, i] = D[i] * R[i, i]

        return R

