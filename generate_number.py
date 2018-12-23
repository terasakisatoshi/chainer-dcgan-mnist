from glob import glob
import os
import time

import chainer
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

from net_mnist import Generator
from train_mnist_classifier import MNISTLeNet

EPSILON = 1e-7


def find_latest(result_dir):
    files = glob(os.path.join(result_dir, 'gen_epoch_*.npz'))
    files = [os.path.basename(f) for f in files]
    files = [os.path.splitext(f)[0].split('_')[-1] for f in files]
    latest = max(list(map(lambda s: int(s), files)))
    return os.path.join(result_dir, 'gen_epoch_{}.npz'.format(latest))


def restore_generator(result_dir):
    generator = Generator(n_hidden=100)
    chainer.serializers.load_npz(find_latest(result_dir), generator)
    return generator


def preview_sample(result_dir, row=10, col=10):
    generator = restore_generator(result_dir)
    xp = generator.xp
    n_images = row * col
    z = chainer.Variable(xp.asarray(generator.make_hidden(n_images)))
    with chainer.using_config('train', False):
        x = generator(z).array
    x = np.asarray(np.clip(x * 255, 0.0, 255.0), dtype=np.uint8)
    x = x.reshape(row, col, 1, 28, 28)
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((row * 28, col * 28))
    preview_path = 'preview.png'
    Image.fromarray(x).save(preview_path)


def generate_number_from_camera(result_dir):
    generator = restore_generator(result_dir)
    # restore classifier
    classifier = chainer.links.Classifier(MNISTLeNet())
    chainer.serializers.load_npz('result/classifier.npz', classifier)

    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print('Error opening video stream or file')
        exit(1)
    fps_time = 0
    while cap.isOpened():
        ret_val, image = cap.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (10, 10))

        zsample = -1 + 2 / (np.max(image) - np.min(image) + EPSILON) * (image - np.min(image))
        zsample = zsample.reshape(1, 100, 1, 1).astype(np.float32)
        with chainer.using_config('trian', False):
            with chainer.using_config('enable_backprop', False):
                number = generator(zsample)
                prob = classifier.predictor(number)
        number = number.array
        prob = prob.array
        number = np.asarray(np.clip(number * 255, 0.0, 255.0), dtype=np.uint8)

        number_image = cv2.resize(
            number.reshape(28, 28).astype(np.uint8),
            (int(1.5 * 448), int(1.5 * 448))
        )

        number_image = cv2.cvtColor(number_image, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            number_image,
            'FPS: % f number: %d' % (1.0 / (time.time() - fps_time), np.argmax(prob)),
            (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow('generate MNIST-like number', number_image)
        fps_time = time.time()
        # press Esc to exit
        if cv2.waitKey(1) == 27:
            break


def main():
    result_dir = 'result/'
    # preview_sample(result_dir)
    generate_number_from_camera(result_dir)


if __name__ == '__main__':
    main()
