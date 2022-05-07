# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from dataset import dataset
from anvae import ANVAE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from IPython import display

import os
from datetime import datetime
from utils_mengfei import plot_zmeans, get_classification_report
from utils_mengfei import plot_true_prediction
from label2params import label2param, get_params

# setting tf gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus: ', gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

DATA_NAME = 'spinodal_28'  # spinodal_28, mnist
RUN_MODE = 'train_anvae'  # train_anvae, train_latent2params
IMG_DIR = 'Output/images_%s/' %(DATA_NAME)
LOG_DIR = './tf_summary/%s/' %(DATA_NAME)
RUN_NAME = str(datetime.now()).split('.')[0].split()[0]
# if RUN_NAME == 'train_latent2params':
#     RUN_NAME = '2021-08-06'  # here make sure you are retrieve the correct model
if DATA_NAME.startswith('spinodal'):
    if DATA_NAME.endswith('reg'):
        TRAIN_PATH = "../data/vtk2D_512to32_2dcontinuous_10k.csv"
    else:
        TRAIN_PATH = "../data/train28_00_new.csv"  # "./Data/mnist_train.csv"
elif DATA_NAME.startswith('mnist'):
    TRAIN_PATH = "../data/train_mnist.csv"  # "./Data/mnist_train.csv"
MODEL_PATH = "save_model/%s/" %(DATA_NAME)

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# RUN_NAME = "Apr15"
# TRAIN_PATH = "../data/train_mnist.csv"  # "./Data/mnist_train.csv"
# TEST_PATH = "../data/test_mnist.csv"  # "./Data/mnist_test.csv"
# MODEL_PATH = "save_model/"  #  "./Models/new/"
USE_SAVED_MODEL = False
SAVED_MODEL_PATH = "save_model/"  # "./load_model/Mar_8_2500-25"
LATENT_SPACES = 3
TRAIN_BUF=60000
BATCH_SIZE=16
TEST_BUF=10000
DIMS = (32,32,1)
N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
N_EPOCHS = 10
LEARNING_RATE_AE = 0.001 
LEARNING_RATE_DISC = 0.001 
LEARNING_RATE_GEN = 0.001 


# Plots the reconstruction of an input
def plot_reconstruction(model, data, epoch, step):
    recon, _, _, _, _ = model(data)
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=(8,4))
    for axi, (dat, lab) in enumerate(
        zip(
            [data, recon],
            ["data", "data recon"],
        )
    ):
        for ex in range(1):
            
            if len(dat[ex].shape) != 3:
                axs[axi].matshow(
                    np.squeeze(dat[ex][0]), cmap=plt.cm.Greys, vmin=0, vmax=1
                )
            else:
                axs[axi].matshow(
                    np.squeeze(dat[ex]), cmap=plt.cm.Greys, vmin=0, vmax=1
                )
            axs[axi].axes.get_xaxis().set_ticks([])
            axs[axi].axes.get_yaxis().set_ticks([])
        axs[axi].set_ylabel(lab)

    # plt.savefig('Output/images/image_{}_{}_{}.png'.format(RUN_NAME, epoch, step))
    plt.savefig('{}/image_{}_{}_{}.png'.format(IMG_DIR, RUN_NAME, epoch, step))


# Samples images from the latent spaces
def plot_sample(image_batch, n, epoch, step):
    fig = plt.figure(figsize=(n, n))
    grid = ImageGrid(fig, 111, nrows_ncols=(n, n), axes_pad=0.1)

    for ax, im in zip(grid, image_batch):
        ax.imshow(np.squeeze(im), cmap=plt.cm.Greys, vmin=0, vmax=1)

    # plt.savefig('Output/images/sample_image_grid_{}_{}_{}.png'.format(RUN_NAME, epoch, step))
    plt.savefig('{}/sample_image_grid_{}_{}_{}.png'.format(IMG_DIR, RUN_NAME, epoch, step))


# Create dataset - handles recieving and shuffling batches
train_data = dataset(data_path = TRAIN_PATH, batch_size=BATCH_SIZE, data_name=DATA_NAME)

# Create training model
train_model = ANVAE(BATCH_SIZE, LEARNING_RATE_AE, LEARNING_RATE_DISC, LEARNING_RATE_GEN)

# Create checkpoint to save training state
checkpoint = tf.train.Checkpoint(model=train_model)

if RUN_MODE == 'train_anvae':

    # If using a saved mode, load it
    if USE_SAVED_MODEL:
        checkpoint.restore(SAVED_MODEL_PATH)

    for epoch in range(N_EPOCHS):
        loss = []
        iteration = 0
        epoch_completed = False

        while epoch_completed == False:
            print("Iteration: {}".format(iteration))
            debug = False
            if (iteration % 100 == 0):
                debug = True

            # Get the next batch of images, labels from dataset
            image_batch, label_batch, epoch_completed = train_data.next_batch()

            # Call the train model step on the current batch of images
            logits, _, _, _, _ = train_model(image_batch)

            iteration += 1

            # Saving model, plotting reconstruction, sampling
            if (train_model.step_count % 200 == 0):

                # Save the current state of the model
                checkpoint.save(MODEL_PATH + "{}_{}".format(RUN_NAME, train_model.step_count))

                print("\nEpoch: {}\n".format(epoch))

                # Clears plot
                display.clear_output()

                # Plot reconstruction
                plot_reconstruction(train_model, image_batch, epoch, iteration)

                # Sample from latent spaces
                sample_image_batch = train_model.sample(4*4, 1)

                # Plot latent space samples
                plot_sample(sample_image_batch, 4, epoch, iteration)
elif RUN_MODE == 'train_latent2params':
    # load anvae model checkpoint
    with open(MODEL_PATH + 'checkpoint', 'r', encoding='utf-8') as f:
        for raw in f:
            name = raw.strip().replace('"', '').replace(' ', '').split(':')[-1]
    print('load checkpoint file : ', name)

    USE_SAVED_MODEL = True
    SAVED_MODEL_PATH = MODEL_PATH + name
    # If using a saved mode, load it
    if USE_SAVED_MODEL:
        checkpoint.restore(SAVED_MODEL_PATH)

    # apply trained encoder to convert images to latent space, then map to parameter space
    data_size = train_data.size()
    train_size = 0.8
    train_data = dataset(data_path=TRAIN_PATH, batch_size=data_size, data_name=DATA_NAME)
    image, label, _ = train_data.next_batch()
    x_train, y_train = image[:int(train_size*data_size)], label[:int(train_size*data_size)]
    x_test, y_test = image[int(train_size*data_size):], label[int(train_size*data_size):]

    # ----------------------------
    # - map to parameters space  -
    # ----------------------------

    x_train_latent = train_model.image2latent(x_train)
    x_test_latent = train_model.image2latent(x_test)
    print('x_train, x_test, x_train_latent, x_test_latent, y_train, y_test\n',
          x_train.shape, x_test.shape, x_train_latent.shape, x_test_latent.shape, y_train.shape, y_test.shape)

    # 3. use trained encoder to do regression for vtk2D_512to32_2dcontinuous_10k.csv
    if DATA_NAME.endswith('reg'):
        y_train_param = y_train
        y_test_param = y_test

        encoder_zmeans = train_model.encoder_zmeans(input_shape=x_train_latent.shape[1:], latent_dim=2)
        history = encoder_zmeans.fit(x_train_latent, y_train_param, epochs=10, batch_size=64,
                                     validation_data=(x_test_latent, y_test_param))
        test_loss, test_acc = encoder_zmeans.evaluate(x_test_latent, y_test)
        print('encoder_zmeans test accuracy: %s', test_acc)
        zmeans = encoder_zmeans.predict(x_test_latent)
        L1_norm_param_predict = str(np.mean(np.abs(zmeans - y_test_param)))
        plot_true_prediction(zmeans, y_test, 'Output/reg_true_prediction')


    # #  1. use trained encoder to do classification
    # encoder_cls = train_model.encoder_cls(input_shape=x_train_latent.shape[1:], num_class=len(np.unique(y_train)))
    # history = encoder_cls.fit(x_train_latent, y_train, epochs=10, batch_size=64,
    #                           validation_data=(x_test_latent, y_test))
    # test_loss, test_acc = encoder_cls.evaluate(x_test_latent, y_test)
    # print('encoder_cls test accuracy: %s', test_acc)
    # y_pred = encoder_cls.predict(x_test_latent)
    # y_pred_01, accuracy, report = get_classification_report(y_test, y_pred)
    # with open(IMG_DIR+'classification_report.txt', 'w', encoding='utf-8') as f:
    #     f.write(report)
    #
    # # 2. use trained encoder to map latent space to param space
    # if DATA_NAME.startswith('spinodal'):
    #     y_train_param = get_params(y_train, label2param)
    #     y_test_param = get_params(y_test, label2param)
    #
    #     encoder_zmeans = train_model.encoder_zmeans(input_shape=x_train_latent.shape[1:], latent_dim=2)
    #     history = encoder_zmeans.fit(x_train_latent, y_train_param, epochs=10, batch_size=64,
    #                                  validation_data=(x_test_latent, y_test_param))
    #     test_loss, test_acc = encoder_cls.evaluate(x_test_latent, y_test)
    #     print('encoder_zmeans test accuracy: %s', test_acc)
    #     zmeans = encoder_zmeans.predict(x_test_latent)
    #     L1_norm_param_predict = str(np.mean(np.abs(zmeans - y_test_param)))
    #     plot_zmeans(zmeans, y_test, IMG_DIR+'parameter_prediction.png')
    #
    #
    #
    #

    
