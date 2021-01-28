import numpy as np
import os
from PIL import Image
import tensorflow as tf 
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt 
import argparse

def get_model_approach_3():
    img_input = tf.keras.Input(shape=(256, ))
    output = Dense(256, activation='sigmoid')(img_input)
    model = tf.keras.Model(img_input, output)
    model.compile(optimizer='sgd', loss='mse') # or binary cross entropy?
    return model


def get_dataset(dir, add_noise):
    filenames = os.listdir(dir)
    filenames.sort()
    dataset_x = []
    dataset_y = []
    array2label = {}


    for i, file in enumerate(filenames):
        img = Image.open(os.path.join(dir, file))
        arr = np.reshape(np.array(img)/255, (256,)).astype('float64').tolist()
        dataset_x.append(arr)
        dataset_y.append(arr)
        array2label[tuple(arr)] = file[4]
    

    if add_noise:
        idx = np.arange(256)
        for i in range(20):
            noise = np.random.normal(0.0, add_noise, 25)
            idx = np.random.permutation(idx)
            for j in range(25):
                dataset_x[i][idx[j]] += noise[j]

    return np.array(dataset_x), np.array(dataset_y), array2label


def split(dataset_x, dataset_y, train, arr2label):
    assert len(dataset_x) == len(dataset_y)
    assert train <= len(dataset_x)
    idx = np.arange(len(dataset_x))
    idx = np.random.permutation(idx, )

    train_x = dataset_x[idx[:train]]
    train_y = dataset_y[idx[:train]]

    test_x = dataset_x[idx[train:]]
    test_y = dataset_y[idx[train:]]

    if train < 20:
        print('train samples:')
        for arr in train_x:
            print(arr2label[tuple(list(arr))], end=' ')
        print('')
        print('test samples:')
        for arr in test_x:
            print(arr2label[tuple(list(arr))], end=' ')
        print('')
    else:
        print('train all samples')


    return train_x, train_y, test_x, test_y




def train_process(img_path, batch, epochs, plot, train, add_noise, verbose):
    model = get_model_approach_3()

    """
    initial model''s weight to diagonal mat
    """
    diag = np.zeros([256, 256])
    bias = np.zeros(256)
    for i in range(256):
        diag[i][i] = 1
    dense_layer = model.layers[1]
    dense_layer.set_weights((diag, bias))
    dense_layer.trainable=False



    dataset_x, dataset_y, arr2label = get_dataset(img_path, add_noise)
    train_x, train_y, test_x, test_y = split(dataset_x, dataset_y, train, arr2label)
    if train < 20:
        history = model.fit(train_x, train_y, batch_size=batch, epochs=epochs, verbose=verbose, validation_data=(test_x, test_y))
    else:
        history = model.fit(train_x, train_y, batch_size=batch, epochs=epochs, verbose=1)
    if plot:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        x = np.arange(epochs)
        plt.plot(x, loss, label='loss')
        plt.plot(x, val_loss, label='val_loss')
        plt.show()

    print('W matrix')
    print(model.layers[1].weights[0])
    print('bias')
    print(model.layers[1].weights[1])
    # test on the total datset.....no matter how the model is trained
    print('test the model on the whole dataset')
    pred = model.predict(dataset_x)
    fhs = []
    fhas = []
    cnt = 0
    for i in range(20):
        y_pred = np.array([0 if pixel < 0.5 else 1 for pixel in pred[i]]).astype('int')
        y_true = dataset_y[i]
        label = arr2label[tuple(list(y_true))]
        y_true = y_true.astype('int')
        
        hit = np.sum((1-y_true) & (1-y_pred))
        total = np.sum(1-y_true)
        pred0 = np.sum(1-y_pred)
        Fh  = hit / total
        Fha  = (pred0-hit) / pred0

        print('label: ',  label, 'Fh: ', format(Fh, '.2f'), 'Fha: ', format(Fha, '.2f'))
        fhs.append(Fh)
        fhas.append(Fha)

    training_samples = []
    for i in range(len(train_x)):
        training_samples.append(arr2label[tuple(list(train_x[i]))])
    

    return training_samples, fhs, fhas


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='Gray')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=5)
    parser.add_argument('-p', '--plot', type=bool, default=True)
    parser.add_argument('-t', '--train', type=int, default=10)
    parser.add_argument('-n', '--noise', type=float, default=0.0)
    parser.add_argument('-v', '--verbose', type=int, default=1)
    return parser.parse_known_args()


if __name__ == '__main__':
    args, _ = arg_parser()
    if args.train > 20:
        print('train numbers should less then total numbers, which is 20')
    else:
        train_process(args.data_path, args.batch_size, args.epochs, args.plot, args.train, args.noise, args.verbose)



