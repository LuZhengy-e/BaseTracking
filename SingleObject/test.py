import os
import sys
import logging
import numpy as np
from copy import deepcopy
from configparser import ConfigParser
from matplotlib import pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking import SingleTracker
from Evaluate.factory import EvaluateFactory

logging.getLogger().setLevel(logging.INFO)


def main(config: ConfigParser):
    circle1_cfg = ConfigParser()
    circle2_cfg = ConfigParser()
    line_cfg = ConfigParser()

    circle1_cfg.read("config/test1.cfg")
    line_cfg.read("config/test2.cfg")
    circle2_cfg.read("config/test3.cfg")
    pd = 0.8

    evaluater1 = EvaluateFactory.get(circle1_cfg)
    ground_truth1 = evaluater1.create(0)

    meas = []
    R = np.identity(2) * float(circle1_cfg.get("Evaluate", "sigma"))

    init_x = np.array(
        [0, 0, np.pi / 2, 200, 6.28]
    )

    # init_x = np.array(
    #     [0, 0, 0, 6.28]
    # )

    shape = init_x.shape[0]
    tracker = SingleTracker(config, init_x, np.ones((shape, shape)))

    for t, pos in ground_truth1.items():
        logging.info(f"-----------time_{t}------------")
        z = np.zeros((3,), dtype=float)
        z[0:2] = np.random.multivariate_normal(ground_truth1[t][0:2], R)

        angle = ground_truth1[t][2]

        meas_angle = np.random.normal(angle, 0.1)
        z[2] = meas_angle

        meas.append(z.copy())

        if t > 0:
            x, P = tracker.predict()
            tracker.update(x=x, P=P, meas=z)

    # plot
    end_time = len(meas)
    plt.figure()

    for t in range(end_time - 1):
        plt.plot([ground_truth1[t][0], ground_truth1[t + 1][0]],
                 [ground_truth1[t][1], ground_truth1[t + 1][1]], c="red")
        plt.plot([meas[t][0], meas[t + 1][0]],
                 [meas[t][1], meas[t + 1][1]], c="black")
        plt.plot([tracker.x[t][0], tracker.x[t + 1][0]],
                 [tracker.x[t][1], tracker.x[t + 1][1]], c="blue")

    evaluater2 = EvaluateFactory.get(line_cfg)
    ground_truth2 = evaluater2.create()

    meas = []
    R = np.identity(2) * float(line_cfg.get("Evaluate", "sigma"))

    init_x = tracker.x[end_time - 1].copy()

    tracker = SingleTracker(config, init_x, tracker.P[end_time - 1].copy())

    for t, pos in ground_truth2.items():
        logging.info(f"-----------time_{t + 100}------------")
        z = np.zeros((3,), dtype=float)
        z[0:2] = np.random.multivariate_normal(ground_truth2[t][0:2], R)

        angle = ground_truth2[t][2]

        meas_angle = np.random.normal(angle, 0.1)
        z[2] = meas_angle

        meas.append(z.copy())

        if t > 0:
            x, P = tracker.predict()
            tracker.update(x=x, P=P, meas=z)

    # plot
    end_time = len(meas)

    for t in range(end_time - 1):
        plt.plot([ground_truth2[t][0], ground_truth2[t + 1][0]],
                 [ground_truth2[t][1], ground_truth2[t + 1][1]], c="red")
        plt.plot([meas[t][0], meas[t + 1][0]],
                 [meas[t][1], meas[t + 1][1]], c="black")
        plt.plot([tracker.x[t][0], tracker.x[t + 1][0]],
                 [tracker.x[t][1], tracker.x[t + 1][1]], c="blue")

    evaluater3 = EvaluateFactory.get(circle2_cfg)
    ground_truth3 = evaluater3.create(3 * np.pi / 2)

    meas = []
    R = np.identity(2) * float(circle2_cfg.get("Evaluate", "sigma"))

    init_x = tracker.x[end_time - 1].copy()

    tracker = SingleTracker(config, init_x, tracker.P[end_time - 1].copy())

    for t, pos in ground_truth3.items():
        logging.info(f"-----------time_{t + 200}------------")
        z = np.zeros((3,), dtype=float)
        z[0:2] = np.random.multivariate_normal(ground_truth3[t][0:2], R)

        angle = ground_truth3[t][2]

        meas_angle = np.random.normal(angle, 0.1)
        z[2] = meas_angle

        meas.append(z.copy())

        if t > 0:
            x, P = tracker.predict()
            tracker.update(x=x, P=P, meas=z)

    # plot
    end_time = len(meas)

    for t in range(end_time - 1):
        plt.plot([ground_truth3[t][0], ground_truth3[t + 1][0]],
                 [ground_truth3[t][1], ground_truth3[t + 1][1]], c="red")
        plt.plot([meas[t][0], meas[t + 1][0]],
                 [meas[t][1], meas[t + 1][1]], c="black")
        plt.plot([tracker.x[t][0], tracker.x[t + 1][0]],
                 [tracker.x[t][1], tracker.x[t + 1][1]], c="blue")

    plt.axis('equal')
    plt.title("CCVInv")
    plt.xlabel(xlabel='x position/m')
    plt.ylabel(ylabel='y position/m')
    plt.savefig("CCVInvMixed.png")

    plt.show()


if __name__ == '__main__':
    np.random.seed(157)
    config = ConfigParser()
    config.read("config/config.cfg")

    main(config)
