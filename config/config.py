import os
import yaml


def load_train_config():
    with open(file="config/train.yaml", mode='r') as f:
        return yaml.load(f, Loader=yaml.loader.SafeLoader)


def load_test_config():
    with open(file="config/test.yaml", mode='r') as f:
        return yaml.load(f, Loader=yaml.loader.SafeLoader)
