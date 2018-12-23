import random

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training, datasets, iterators, optimizers
from chainer.training import extensions
import numpy as np
from tqdm import tqdm

DEVICE = 0
BATCH_SIZE = 128


class MNISTLeNet(chainer.Chain):

    def __init__(self):
        super(MNISTLeNet, self).__init__()
        with self.init_scope():
            ksize = 5
            self.c1 = L.Convolution2D(None, 20, ksize=ksize, pad=ksize // 2)
            self.c2 = L.Convolution2D(None, 50, ksize=ksize, pad=ksize // 2)
            self.l1 = L.Linear(None, 500)
            self.l2 = L.Linear(None, 10)

    @chainer.static_graph
    def __call__(self, x):
        h = F.relu(self.c1(x))
        h = F.max_pooling_2d(h, ksize=(2, 2))
        h = F.relu(self.c2(h))
        h = F.relu(self.l1(h))
        h = self.l2(h)
        return h


def train():
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    model = L.Classifier(MNISTLeNet())
    if DEVICE >= 0:
        chainer.cuda.get_device_from_id(DEVICE).use()
        chainer.cuda.check_cuda_available()
        model.to_gpu()
    train, test = chainer.datasets.get_mnist(ndim=3)
    train_iter = iterators.SerialIterator(train, BATCH_SIZE, shuffle=True)
    test_iter = iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)

    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, device=DEVICE)
    trainer = training.Trainer(updater, (12, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.Evaluator(test_iter, model, device=DEVICE))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    chainer.serializers.save_npz('result/classifier.npz', model)


def predict():
    model = L.Classifier(MNISTLeNet())
    chainer.serializers.load_npz('result/classifier.npz', model)
    train, test = chainer.datasets.get_mnist(ndim=3)
    counter = 0
    acc = 0
    with chainer.using_config('trian', False):
        with chainer.using_config('enable_backprop', False):
            for t in tqdm(test):
                counter += 1
                x, ans = t
                result = model.predictor(np.array([x])).data[0]
                if ans == np.argmax(result):
                    acc += 1
    print(acc / counter)


def main():
    train()
    predict()


if __name__ == '__main__':
    main()
