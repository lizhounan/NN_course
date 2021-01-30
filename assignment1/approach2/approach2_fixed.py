import numpy as np
import os
from PIL import Image
import tensorflow as tf 
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt 
import argparse

def get_model_approach_2():
    img_input = tf.keras.Input(shape=(256, ))
    output = Dense(20, activation='softmax')(img_input)
    model = tf.keras.Model(img_input, output)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def get_dataset(dir, add_noise):
    filenames = os.listdir(dir)
    filenames.sort()
    dataset_x = []
    dataset_y = []
    onehot2label = {}
    index2label = {}

    # get Fh and Fha table by the way
    Fh = {}
    Fha = {}

    for i, file in enumerate(filenames):
        img = Image.open(os.path.join(dir, file))
        arr = np.reshape(np.array(img)/255, (256,)).astype('float64').tolist()
        dataset_x.append(arr)
        label = [0] * 20
        label[i] = 1
        dataset_y.append(label)
        label = tuple(label)
        onehot2label[label] = file[4]
        index2label[i] = file[4]
    
    # get Fh and Fha
    for i in range(20):
        for j in range(20):
            f1 = filenames[i][4]
            f2 = filenames[j][4]
            arr1 = np.array(dataset_x[i]).astype('int')
            arr2 = np.array(dataset_x[j]).astype('int')
            hit = np.sum((1-arr1) & (1-arr2))
            total = np.sum(1-arr1)
            pred = np.sum(1-arr2)
            Fh[f1+f2] = hit / total
            Fha[f1+f2] = (pred-hit) / pred

    if add_noise:
        idx = np.arange(256)
        for i in range(20):
            noise = np.random.normal(0.0, add_noise, 25)
            idx = np.random.permutation(idx)
            for j in range(25):
                dataset_x[i][idx[j]] += noise[j]
    dataset_x = np.array(dataset_x)
    min_ = np.min(dataset_x, keepdims=True)
    max_ = np.max(dataset_x, keepdims=True)
    dataset_x = (dataset_x - min_) / max_
    return dataset_x, np.array(dataset_y), onehot2label, index2label, Fh, Fha


def split(dataset_x, dataset_y, train, onehot2label):
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
        for arr in train_y:
            print(onehot2label[tuple(list(arr))], end=' ')
        print('')
        print('test samples:')
        for arr in test_y:
            print(onehot2label[tuple(list(arr))], end=' ')
        print('')
    else:
        print('train all samples')


    return train_x, train_y, test_x, test_y


def split_fixed(dataset_x, dataset_y):
    assert len(dataset_x) == len(dataset_y)
    train_idx = [18, 13, 14, 11, 6, 12, 10, 7, 5, 3]
    test_idx = [0, 1, 2 ,4 ,8, 9, 15, 16, 17, 19]
    train_x = dataset_x[train_idx]
    train_y = dataset_y[train_idx]
    test_x = dataset_x[test_idx]
    test_y = dataset_y[test_idx]
    return train_x, train_y, test_x, test_y




def train_process(img_path, batch, epochs, plot, train, add_noise, verbose):
    model = get_model_approach_2()
    dataset_x, dataset_y, onehot2label, index2label, Fh, Fha = get_dataset(img_path, add_noise)
    train_x, train_y, test_x, test_y = split_fixed(dataset_x, dataset_y)
    if train < 20:
        history = model.fit(train_x, train_y, batch_size=batch, epochs=epochs, verbose=verbose, validation_data=(test_x, test_y))
    else:
        history = model.fit(train_x, train_y, batch_size=batch, epochs=epochs, verbose=1)
    if plot:
        acc = history.history['accuracy']
        loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        x = np.arange(epochs)
        plt.plot(x, acc, label='acc')
        plt.plot(x, loss, label='loss')
        plt.plot(x, val_acc, label='val_acc')
        plt.plot(x, val_loss, label='val_loss')
        plt.legend()
        plt.show()

    
    fhs = []
    fhas = []
    
    print('now test the model on the train set')
    pred = model.predict(train_x)
    cnt = 0
    for i in range(10):
        y_pred = index2label[np.argmax(pred[i])]
        y_true = onehot2label[tuple(list(train_y[i]))]
        print('truth: ',  y_true, '  pred: ', y_pred, 'Fh: ', format(Fh[y_true+y_pred], '.2f'), 'Fha: ', format(Fha[y_true+y_pred], '.2f'))
        fhs.append(Fh[y_true+y_pred])
        fhas.append(format(Fha[y_true+y_pred]))
        if y_pred == y_true:
            cnt += 1

    print('acc: ', cnt / 10)

    print('now test the model on the test set')
    pred = model.predict(test_x)
    cnt = 0
    for i in range(10):
        y_pred = index2label[np.argmax(pred[i])]
        y_true = onehot2label[tuple(list(test_y[i]))]
        print('truth: ',  y_true, '  pred: ', y_pred, 'Fh: ', format(Fh[y_true+y_pred], '.2f'), 'Fha: ', format(Fha[y_true+y_pred], '.2f'))
        fhs.append(Fh[y_true+y_pred])
        fhas.append(format(Fha[y_true+y_pred]))
        if y_pred == y_true:
            cnt += 1

    print('acc: ', cnt / 10)

    # # test on the total datset.....no matter how the model is trained
    # print('test the model on the whole dataset')
    # pred = model.predict(dataset_x)
    # fhs = []
    # fhas = []
    # cnt = 0
    # for i in range(20):
    #     y_pred = index2label[np.argmax(pred[i])]
    #     y_true = onehot2label[tuple(list(dataset_y[i]))]
    #     print('pred: ', y_pred, '  truth: ',  y_true, 'Fh: ', format(Fh[y_true+y_pred], '.2f'), 'Fha: ', format(Fha[y_true+y_pred], '.2f'))
    #     fhs.append(Fh[y_true+y_pred])
    #     fhas.append(format(Fha[y_true+y_pred]))
    #     if y_pred == y_true:
    #         cnt += 1

    # print('acc: ', cnt / 20)

    ordered_samples = []
    for i in range(10):
        ordered_samples.append(onehot2label[tuple(list(train_y[i]))])

    for i in range(10):
        ordered_samples.append(onehot2label[tuple(list(test_y[i]))])
    

    return ordered_samples, fhs, fhas


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, default='Gray')
    parser.add_argument('-e', '--epochs', type=int, default=100)
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



