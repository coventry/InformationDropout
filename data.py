import gzip
from pickle import load

DATA = None


def get_training_data():
    global DATA
    if DATA is None:
        DATA = load(gzip.open('./artifacts/train-tensor.pkl.gz'))
    return DATA
