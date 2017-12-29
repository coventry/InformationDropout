"""Implementation of Bentes et.al 2016, "Ship-Iceberg Discrimination with
Convolutional Neural Networks in High Resolution SAR Images", using Information
Dropout.

"""

import numpy as np
from pprint import pprint

from random import Random
import torch as T
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim
from torch.nn import Module, ModuleList, Conv2d, MaxPool2d, ZeroPad2d, Linear

from data import get_training_data
from information_dropout import InformationDropoutLayer, InfoActivations

if T.cuda.is_available():  # Define TP for accessing GPU-agnostic operations
    import torch.cuda as TP
else:
    import torch as TP


def argmax(t):
    "The coordinates of the max value in tensor t"
    return np.unravel_index(t.view(-1).max(0)[1].data.cpu(), t.size())


def print_max_nbd(t):
    "Print values in a neighborhood of the max value in t"
    x, y = [v[0] for v in argmax(t)]
    print(f'max coords at {(x, y)}')
    print(t[max(0, x - 3): min(x + 3, t.size(0)),
            max(0, y - 3): min(y + 3, t.size(0))])


class Flatten(Module):
    "Simply flattens all but the batch (first) dimension of the input"

    def forward(self, i):
        return i.view(i.size(0), -1)


TP.manual_seed(0)  # Note: Messing with the current RNG state.

# Shape arguments passed to `InformationDropoutLayer`'s.
INFO_ARGS = [dict(output_size=(32, 38, 38), in_channels=2, out_channels=32,
                  kernel_size=2, stride=2, max_alpha=0.),  # 0
             dict(output_size=(64, 10, 10), in_channels=32, out_channels=64,
                  kernel_size=2, stride=2, max_alpha=0.),  # 1
             ]


def make_info_layer(idx):
    return (INFO_ARGS[idx]['output_size'],
            InformationDropoutLayer(Conv2d, **INFO_ARGS[idx]))


SHAPES_LAYERS = [  # (output_shape, layer) tuples. Cross-checked in `.forward`.
    ((2, 75, 75), None),  # Input layer
    ((2, 76, 76), ZeroPad2d((0, 1, 0, 1))),
    make_info_layer(0),
    # ((32, 36, 72), ZeroPad2d((0, 1, 0, 1))),
    ((32, 19, 19), MaxPool2d(kernel_size=2, stride=2)),
    ((32, 20, 20), ZeroPad2d((0, 1, 0, 1))),
    make_info_layer(1),
    ((64, 5, 5), MaxPool2d(kernel_size=2, stride=2)),
    ((1600,), Flatten()),
    ((2,), Linear(in_features=1600, out_features=2)),
]


class BentesModel(Module):
    """Implementation of Bentes et.al 2016"""

    def __init__(self) -> None:
        super(BentesModel, self).__init__()
        # Shapes taken from Fig. 2, p. 2 of Bentes et al., with zero padding to
        # make the kernels line up.
        self.shapes_layers = SHAPES_LAYERS
        # Add the layers as a ModuleList, to register their parameters
        self.layers = ModuleList([t[1] for t in self.shapes_layers[1:]])

    def forward(self, i: Variable) -> InfoActivations:
        """Computes log (ship, iceberg) probabilities, and KL-divergence of posterior
        from prior.

        Args:
            `i`: A (Undetermined, 2, 75, 75) tensor containing the polarized
                 radar images, one in each channel.

        Returns:
            `InfoActivations` instance. `activations` field is a 2-tensor for
            the log probability that input is from a ship or an iceberg. `kl`
            field contains the total KL divergence of the posteriors in each
            `InformationDropoutLayer` from the prior.

        """
        kl = Variable(T.zeros(i.size()[0]).type(TP.FloatTensor))
        assert i.size()[1:] == self.shapes_layers[0][0]
        output = i
        for size, layer in self.shapes_layers[1:]:
            output = layer(output)
            if isinstance(output, InfoActivations):
                kl = kl + output.kl  # Gather KL divergences
                output = output.activations  # Prepare output for next layer
            assert output.size()[1:] == size
        return InfoActivations(activations=output, kl=kl)


def train():
    PYRNG = Random(0)
    ttv_proportions = dict(test=0.001, train=.96, validation=0.039)
    # Whiten, add flips, mask regions outside circle, train/test/val split
    DATA = (get_training_data().to_gpu().normalize().enrich().mask_circle().
            test_train_validation(PYRNG, **ttv_proportions))
    VALDATA = DATA.validation.get_examples(50, PYRNG)
    VALIMGS = Variable(T.from_numpy(VALDATA.images).type(TP.FloatTensor))
    VALCLASSES = Variable(TP.LongTensor(VALDATA.is_iceberg))
    BETA = 1e1
    BETA_FACTOR = .9999
    BATCH_SIZE = 32
    model = BentesModel()
    if T.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)
    for i in range(1_000_000_000):
        scheduler.step()
        optimizer.zero_grad()
        batch = DATA.train.get_examples(BATCH_SIZE, PYRNG).rotate(PYRNG)
        imgvar = Variable(T.from_numpy(batch.images).type(TP.FloatTensor))
        result = model(imgvar)
        classvar = Variable(TP.LongTensor(batch.is_iceberg))
        accuracy = F.cross_entropy(result.activations, classvar)
        kl = T.mean(result.kl)
        loss = accuracy + BETA * kl
        loss.backward()
        valresult = model(VALIMGS)
        valaccuracy = F.cross_entropy(valresult.activations, VALCLASSES)
        optimizer.step()
        gf = lambda t: f'{t.data[0]:12.3f}'  # noqa: E731
        print(f'Step: {i:6d} CE: {gf(accuracy)} KL: {gf(kl)} loss: {gf(loss)} '
              f'val: {gf(valaccuracy)}')
        scores = (F.log_softmax(result.activations, dim=1)
                  .data.cpu().numpy()[list(range(BATCH_SIZE)),
                                      batch.is_iceberg])
        print(np.array(list(zip(*(
            s.astype(float) for s in np.histogram(scores))))).T)
        probs = F.softmax(result.activations).data.cpu().numpy().tolist()
        pprint(list(zip(batch.is_iceberg, probs)))
        BETA *= BETA_FACTOR
        print('first layer parameters/gradients for first kernel')
        print('convolution')
        print(model.layers[1].layer.weight[0])
        print(model.layers[1].layer.weight.grad[0])
        print('noise')
        print(model.layers[1].noise.weight[0])
        print(model.layers[1].noise.weight.grad[0])
        print('prior mean')
        print(model.layers[1].prior.mean[0])
        print(model.layers[1].prior.mean.grad[0])
        print('prior alpha')
        print(model.layers[1].prior.alpha[0])
        print(model.layers[1].prior.alpha.grad[0])


if __name__ == '__main__':
    train()


################
# Open questions
################
#
# 1) Why are the convolutional parameters going to zero?
#
#    This is happening because the KL divergence is easier to push to zero than
#    learning the actual task. The objective function favors sending the
#    weights to zero, and using the biases and noise to fit to the prior. By
#    the time the KL divergence is small enough that the cross entropy obective
#    should dominate, the weights have no gradient because they're so close to
#    zero.
#
#    Could maybe prevent this by making BETA smaller.
#
# 2) Why is there so little variation in the prior?
#
#    That's an illusion. It only looks that way because torch only prints out
#    the corners, where I've zeroed out the data.
#
#    The prior stddev actually drops to .7. The mean does come out constant,
#    but that's not too surprising.
