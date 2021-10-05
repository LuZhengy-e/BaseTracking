import os
import sys
from configparser import ConfigParser

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from filter.predictor import CCVPredictor, CCVInvPredictor, CVPredictor
from filter.updater import LinearUpdater


class PredictFactory:
    factory = {
        "CCV": CCVPredictor,
        "CCVInv": CCVInvPredictor,
        "CV": CVPredictor
    }

    @classmethod
    def get(cls, cfg: ConfigParser):
        predictor_name = cfg.get("Predict", "predictor")
        predictor = cls.factory[predictor_name]

        return predictor(cfg)


class UpdateFactory:
    factory = {
        "Linear": LinearUpdater
    }

    @classmethod
    def get(cls, cfg: ConfigParser):
        updater_name = cfg.get("Update", "updater")
        updater_name = updater_name.split("_")

        updater = cls.factory.get(updater_name[0])

        return updater(cfg)


if __name__ == '__main__':
    config = ConfigParser()
    config.read("config/config.cfg")
    print(UpdateFactory.get(config))
