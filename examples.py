"""Keeps track of registered lists of example data.

If `e` is an `Examples` instance, `id` the string ID for some example, then,
`e.get_example(id)` returns the data for that example. If `idx` is an integer
index into the examples, `e.get_example(idx)` is the `idx`th example.

"""

import attr
import numpy as np
from random import Random
from skimage.transform import rotate
import torch as T
from typing import Iterable, List, Union, Set, Callable  # noqa: F401

from mypy_shim import MyPyShim

np_array_or_tensor = Union[np.ndarray, T.tensor._TensorBase]


class NO_DATA:
    "Used to signal that a field is empty"


@attr.s(frozen=True, slots=True)
class Examples(MyPyShim):

    """Keeps track of registered lists of example data.

    If `e` is an `Examples` instance, `id` the string ID for some example,
    then, `e.get_example(id)` returns the data for that example. If `idx` is an
    integer index into the examples, `e.get_example(idx)` is the `idx`th
    example.

    """
    inclinations:  List[Union[float, NO_DATA]] = attr.ib()  # angle of image
    is_iceberg: List[bool] = attr.ib()                      # True: "iceberg"
    images: np_array_or_tensor = attr.ib()                  # (n, 2, 75, 75)
    ids: List[str] = attr.ib()                              # String IDs

    def __attrs_post_init__(self):
        "Check values are consistent, and construct id -> position index"
        assert len(set(map(len, attr.asdict(self).values()))) == 1, \
            'All entries should be the same length'
        assert set(self.is_iceberg).issubset(set([True, False]))
        assert all(isinstance(a, float) or
                   # Just check the name, not the class, in case we reload.
                   (getattr(a, '__name__', None) == 'NO_DATA')
                   for a in self.inclinations)
        assert self.images.shape[1:] == (2, 75, 75)
        if isinstance(self.images, np.ndarray):
            assert self.images.dtype in (np.float64, np.float32)
        else:
            assert 'FloatTensor' in self.images.type()

    def size(self) -> int:
        return len(self.ids)

    def get_examples_by_idx(self, idxs):
        rv = {k: [v[i] for i in idxs] for k, v in attr.asdict(self).items()}
        rv['images'] = np.stack(rv['images'])  # Convert images back to numpy
        return attr.evolve(self, **rv)

    def get_examples(self, n: int, rng: Random) -> 'Examples':
        "Get a batch of `n` random examples, using `rng` for stochasticity."
        return self.get_examples_by_idx(rng.sample(range(self.size()), n))

    def test_train_validation(self, rng: Random, *dummy,
                              test: float=0., train: float=0.,
                              validation: float=0.) -> 'DataSplit':
        """Return a test/train/validation split of the given proportions.

        Args:
            `rng`: A Random instance to generate stochasticity from.
            `test`, `train`, `validation`: What fraction to assign to each part
                of the split. Must sum to 1.

        """
        idxs = list(range(self.size()))  # idxs[:size*test] is test split, etc.
        rng.shuffle(idxs)
        # This funny construction avoids errors due to order-dependence.
        nms = [a.name for a in attr.fields(DataSplit)]
        args = locals()  # Need to get this outside the comprehension context
        bounds = np.array([0.] + [args[n] for n in nms]).cumsum()
        assert bounds[1:].min() >= 0, 'Split fractions must be non-negative'
        assert np.isclose(bounds[-1], 1), 'Inputs must sum to 1'
        bounds = (bounds * self.size()).astype(int)
        # {split_name: (start_of_split_in_idxs, end_of_split_in_idxs)}
        bounds = {n: (s, e) for n, (s, e) in zip(nms, zip(bounds, bounds[1:]))}
        # Gets the examples referenced by idxs between bounds s and e.
        egs = lambda s, e: self.get_examples_by_idx(idxs[s: e])  # noqa: E731
        rv = {n: egs(s, e) for n, (s, e) in bounds. items()}
        # Check that the splits are disjoint union of current examples
        assert sum(eg.size() for eg in rv.values()) == self.size()
        ids = frozenset(sum((eg.ids for eg in rv.values()), []))
        assert ids == frozenset(self.ids)
        return DataSplit(**rv)

    def normalize(self) -> 'Examples':
        """Normalize each image to mean 0, stddev 1"""
        flat_images = self.images.view(self.size(), -1, 1, 1, 1)
        stddevs = flat_images.std(1)
        assert stddevs.min() > 1e-4
        images = (self.images - flat_images.mean(1)) / stddevs
        allclose = lambda a, v: np.isclose(a, v, atol=1e-5).all()
        assert allclose(images.view(images.size(0), -1).mean(1), 0)
        assert allclose(images.view(images.size(0), -1).std(1), 1)
        return attr.evolve(self, images=images)

    def enrich(self)-> 'Examples':
        "Add flips and rotations of all examples"
        # Identity and vertical flips
        trans = {'': self.images, 'flip-': np.flip(self.images, axis=1)}
        images = np.concatenate(list(trans.values()))
        # Add transform info to the id fields
        ids = [f'{i}-{n}' for i in self.ids for n in trans.keys()]
        updates = dict(ids=list(ids), images=images)  # Updated arguments
        # Duplicate the other fields as needed
        other_fields = set(attr.asdict(self).keys()) - set(updates.keys())
        others = {n: len(trans) * getattr(self, n) for n in other_fields}
        return attr.evolve(self, **updates, **others)

    def mask_circle(self) -> 'Examples':
        "Mask points more than width/2 units from image center."
        h, w = self.images.shape[-2:]
        assert h == w, 'Height and width must match'
        radius = h / 2
        X, Y = np.ogrid[:h, :w]  # Col & row vec, broadcasts to square matrix
        # Center is (radius, radius). Mask points more than radius units away.
        mask = (((X - radius)**2 + (Y - radius)**2) < radius**2)
        ids = [f'{i}-masked' for i in self.ids]
        return attr.evolve(self, images=self.images*mask, ids=ids)

    def rotate(self, rng: Random) -> 'Examples':
        "Rotate each image by a random amount."
        angles = [rng.uniform(0, 360) for _ in self.images]
        images = np.stack([  # Rotate both channels in each image
            np.array([rotate(j, a, preserve_range=True) for j in i])
            for i, a in zip(self.images, angles)])
        ids = [f'{i}-rot-{a:.3f}-degrees' for i, a in zip(self.ids, angles)]
        return attr.evolve(self, images=images, ids=ids)

    def to_gpu(self) -> 'Examples':
        "Send to GPU, if available"
        images = self.images
        if isinstance(images, np.ndarray):
            images = T.from_numpy(self.images)  # Uses same backing store
        ttype = T.cuda.FloatTensor if T.cuda.is_available() else T.FloatTensor
        return attr.evolve(self, images=images.type(ttype))


@attr.s(frozen=True, slots=True)
class DataSplit(MyPyShim):

    test: Examples = attr.ib()
    train: Examples = attr.ib()
    validation: Examples = attr.ib()
