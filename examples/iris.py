# Requires scikit-learn

from convnetpy.vol import Vol
from convnetpy.net import Net
from convnetpy.trainers import Trainer

import random
from sklearn.datasets import load_iris

iris_data = None
network = None
sgd = None
N_TRAIN = 100


def load_data():
    global iris_data

    data = load_iris()

    xs = data.data
    ys = data.target

    inputs = [Vol(list(row)) for row in xs]
    labels = list(ys)

    iris_data = list(zip(inputs, labels))
    random.shuffle(iris_data)
    print('Data loaded...')


def start():
    global network, sgd

    layers = []
    layers.append({'type': 'input', 'out_sx': 1, 'out_sy': 1, 'out_depth': 4})
    layers.append({'type': 'softmax', 'num_classes': 3})  # svm works too
    print('Layers made...')

    network = Net(layers)
    print('Net made...')
    print(network)

    sgd = Trainer(network, {'learning_rate': 0.01, 'momentum': 0.9, 'l2_decay': 0.001, 'batch_size': 1})
    print('Trainer made...')
    print(sgd)


def train():
    global iris_data, sgd

    print('In training...')
    print('iter\ttime\t\tloss\t\ttraining accuracy')
    print('----------------------------------------------------')
    for x, y in iris_data[:N_TRAIN]:
        stats = sgd.train(x, y)
        print('%s\t%.8f\t%.8f\t%.8f' % (str(stats['k']).rjust(6), stats['time'], stats['loss'], stats['accuracy']))


def test():
    global iris_data, network

    print('In testing...')
    right = 0
    for x, y in iris_data[N_TRAIN:]:
        network.forward(x)
        right += network.getPrediction() == y
    accuracy = float(right) / (150 - N_TRAIN) * 100
    print(accuracy)


def main():
    load_data()
    start()
    for i in range(10):
        print('Epoch %d' % i)
        train()
    test()


if __name__ == '__main__':
    main()
