"""Extract training data to single file per example.

E.g., `python extract.py`.

Expects a file `./train.json.7z` (`DATA_PATH`) in the same directory.

Creates files in `./artifacts/train`. (`SAVE_DIR`). They are gzipped pickle
files with paths `<IS_ICEBERG>-<INCLINATION_ANGLE>-<ID>.gz`, where `IS_ICEBERG`
is `1` if it's an example of an iceberg, `0` otherwise, `INCLINATION_ANGLE` is
the angle of the view from the horizon(?), and `ID` is an identification label.
These are all extracted from the original data. This is configured by variable
`SPATH`.

Pickles are in the following form:

    In [1]: {k: type(v) for k, v in d.items()}
    Out[1]: {'id': str, 'image': numpy.ndarray, 'inc_angle': str,
             'is_iceberg': int}

Here `id`, `inc_angle` and `is_iceberg` have the same meaning as their cognates
in the file name, and `image` is the stacked pair of 75x75 images corresponding
to `band_1` and `band_2` in the original data.

    In [2]: d['image'].shape
    Out[2]: (2, 75, 75)
    In [3]: d['image'].dtype
    Out[3]: dtype('float64')

Note that in some cases, the inclination is not given, in which case the
'inc_angle' is a `str`.  Otherwise, it's a float.

"""

from functools import partial
import numpy as np
from pickle import dump
import os
import re
import json

SAVE_DIR = './artifacts/train/'  # Where to save the processed data
os.makedirs(SAVE_DIR, exist_ok=True)

# Format in which to save the data
SPATH = '{is_iceberg}-{inc_angle}-{id}.pkl'
DATA_PATH = './train.json.7z'
assert os.popen('cksum ' + DATA_PATH).read().split()[0] == '553058160', \
        'Training data has changed!'

TRAINING_PATH = 'data/processed/train.json'  # Path to original training data

# https://superuser.com/questions/148498/7zip-how-to-extract-to-std-output
EXTRACT_CMD = "7zr e -so {DATA_PATH} {TRAINING_PATH} | tee".format(
        DATA_PATH=DATA_PATH, TRAINING_PATH=TRAINING_PATH)


def parse_object(o):
    "Sanity check JSON object from data, and convert to python dict"
    assert [[m.start() for m in re.finditer(c, o)] for c in '{}'] == \
        [[0], [len(o) - 1]], "Should only be one set of braces"
    return json.loads(o)


def json_objects():
    "Generator yielding JSON objects in data, as python dictionaries"
    buff = []
    f = os.popen(EXTRACT_CMD)
    assert f.read(1) == '['  # Start of big list of objects
    # Read in data 1MB at a time, until data is fully processed
    for chunk in iter(partial(f.read, 2**20), ''):
        objportions = re.split('}(, *)?', chunk)[::2]  # 2 skips paren-groups
        for objportion in objportions[:-1]:
            yield parse_object(''.join(buff) + objportion + '}')
            buff = []  # Just used everything in the buffer, so empty it
        buff.append(objportions[-1])  # unused portion to include in next chunk
    assert buff == [']']  # End of big list of objects


for d in json_objects():
    bands = [1, 2]
    image = np.array([d['band_{i}'.format(i=i)] for i in bands])
    image = image.reshape(2, 75, 75)
    for i in bands:  # Remove bands and replace with numpy array
        del d['band_{i}'.format(i=i)]
    d.update(image=image)
    dump(d, open(os.path.join(SAVE_DIR, SPATH.format(**d)), 'wb'))
    print(d['id'])

# Compress the saved files
assert not os.system('cd {SAVE_DIR} && echo *.pkl | xargs -P 4 gzip'
                     .format(SAVE_DIR=SAVE_DIR))

# Make these files read-only
assert not os.system('chmod -R a-w {SAVE_DIR} {DATA_PATH}'.format(
        SAVE_DIR=SAVE_DIR, DATA_PATH=DATA_PATH))
