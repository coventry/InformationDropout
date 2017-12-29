"""Extract the examples data as tensors.

E.g. `python preprocessing/single_tensors.py`

Assumes `extract.py` has already been run, producing files for each example in
`./artifacts/train`

"""

from collections import defaultdict
import glob
import gzip
import numpy as np
import os
from pickle import load, dump
import random

from examples import NO_DATA, Examples


def tensors_from_files(savepath):
    """Examples(inclinations, images, ids, is_iceberg) from files under
    `savepath`.

    If it's test data, there's no is_iceberg, so the is_iceberg row for that
    example is `NO_DATA`.

    Some train data has `inc_angle` value `na`. Those `inclination` for those
    rows are also set to `NO_DATA`

    """
    examples = glob.glob(os.path.join(savepath, '*.pkl.gz'))
    random.shuffle(examples)
    results = defaultdict(list)
    for path in examples:
        print(path)
        d = load(gzip.open(path))
        for k, v in d.items():
            results[k].append(v)
        if 'is_iceberg' not in d:
            results['is_iceberg'].append(NO_DATA)
        if isinstance(d['inc_angle'], str):
            assert d['inc_angle'] == 'na'
            results['inc_angle'][-1] = NO_DATA
    return Examples(inclinations=results['inc_angle'],
                    images=np.array(results['image']),
                    ids=results['id'],
                    is_iceberg=results['is_iceberg'])


def main():
    random.seed(0)
    train_egs = tensors_from_files('./artifacts/train')
    dump(train_egs, open('./artifacts/train-tensor.pkl', 'wb'))
    del train_egs
    assert not os.system('gzip ./artifacts/train-tensor.pkl')


if __name__ == '__main__':
    main()
