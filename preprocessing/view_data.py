import gzip
import matplotlib.pyplot as plt
import os
from pickle import load
import random

random.seed(0)

TRAINDIR = './artifacts/train/'
SAVEDIR = './artifacts/images'
os.makedirs(SAVEDIR, exist_ok=True)
SPATH = '{is_iceberg}-{inc_angle}-{id}.png'

examples = os.listdir(TRAINDIR)
random.shuffle(examples)

for p in examples[:]:
    print(p)
    d = load(gzip.open(os.path.join(TRAINDIR, p)))
    plt.clf()
    plt.suptitle('{image_type}, inclination {incl}'.format(
        image_type=('Iceberg' if d['is_iceberg'] else 'ship'),
        incl=d['inc_angle']), fontsize=36)
    for img in [0, 1]:
        plt.subplot(1, 2, img + 1)
        plt.imshow(d['image'][img])
    plt.savefig(os.path.join(SAVEDIR, SPATH.format(**d)))
