import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from configparser import ConfigParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import Circle, Line


class EvaluateFactory:
    factory = {"circle": Circle,
               "line": Line}

    @classmethod
    def get(cls, cfg: ConfigParser):
        evaluate_name = cfg.get("Evaluate", "type")
        evaluater = cls.factory.get(evaluate_name)

        return evaluater(cfg)


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config/config.cfg")

    evaluater = EvaluateFactory.get(config)
    gt = evaluater.create()
    end_time = int(config.get("Evaluate", "end_time"))

    plt.figure()

    for t in range(end_time - 1):
        plt.plot([gt[t][0], gt[t+1][0]], [gt[t][1], gt[t+1][1]], c="red")

    plt.axis('equal')
    plt.show()
