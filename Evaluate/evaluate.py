import numpy as np
from configparser import ConfigParser


class BaseShape:
    def __init__(self, cfg: ConfigParser):
        self.cfg = cfg

    def create(self, **kwargs):
        raise NotImplementedError


class Circle(BaseShape):
    def __init__(self, cfg: ConfigParser):
        super(Circle, self).__init__(cfg)
        self.radius = float(cfg.get("Evaluate", "radius"))
        self.vt = float(cfg.get("Evaluate", "vt"))
        center = cfg.get("Evaluate", "center").split(",")
        self.center = list(map(float, center))
        self.end_time = int(cfg.get("Evaluate", "end_time"))

    def create(self, init_angle=0):
        gt = {}
        angle = init_angle
        omega = self.vt / self.radius

        gt[0] = np.array(
            [self.center[0] + np.cos(angle) * self.radius,
             self.center[1] + np.sin(angle) * self.radius,
             angle + np.pi / 2]
        )

        for t in range(1, self.end_time):
            angle += omega
            real_angle = angle + np.pi / 2
            if angle > 3 * np.pi / 2:
                real_angle = real_angle - 3 * np.pi / 2

            gt[t] = np.array(
                [self.center[0] + np.cos(angle) * self.radius,
                 self.center[1] + np.sin(angle) * self.radius,
                 real_angle]
            )

        return gt


class Line(BaseShape):
    def __init__(self, cfg: ConfigParser):
        super(Line, self).__init__(cfg)
        self.vt = float(cfg.get("Evaluate", "vt"))
        start = cfg.get("Evaluate", "start").split(",")
        self.start = list(map(float, start))
        self.end_time = int(cfg.get("Evaluate", "end_time"))
        self.angle = float(cfg.get("Evaluate", "angle")) * np.pi / 180

    def create(self):
        gt = {}
        x = np.array(
            self.start + [self.angle]
        )
        gt[0] = x.copy()

        for t in range(1, self.end_time):
            x[0:2] = x[0:2] + self.vt * np.array([np.cos(self.angle), np.sin(self.angle)])
            gt[t] = x.copy()

        return gt
