# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 18:17:17 2020

@author: Octavian
"""
import tensorflow as tf
import numpy as np
from dataset import dataset
from anvae import ANVAE
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pandas as pd
from IPython import display

RUN_NAME = "Apr15"
TRAIN_PATH = "../data/train_mnist.csv"  # "./Data/mnist_train.csv"
TEST_PATH = "../data/test_mnist.csv"  # "./Data/mnist_test.csv"
MODEL_PATH = "save_model/"  #  "./Models/new/"
USE_SAVED_MODEL = False
SAVED_MODEL_PATH = "save_model/"  # "./load_model/Mar_8_2500-25"
LATENT_SPACES = 3
TRAIN_BUF=60000
BATCH_SIZE=16
TEST_BUF=10000
DIMS = (32,32,1)
N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
N_EPOCHS = 200
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

    plt.savefig('Output/images/image_{}_{}_{}.png'.format(RUN_NAME, epoch, step))
    
# Samples images from the latent spaces
def plot_sample(image_batch, n, epoch, step):
    fig = plt.figure(figsize=(n, n))
    grid = ImageGrid(fig, 111, nrows_ncols=(n, n), axes_pad=0.1)

    for ax, im in zip(grid, image_batch):
        ax.imshow(np.squeeze(im), cmap=plt.cm.Greys, vmin=0, vmax=1)

    plt.savefig('Output/images/sample_image_grid_{}_{}_{}.png'.format(RUN_NAME, epoch, step))

# Create dataset - handles recieving and shuffling batches
train_data = dataset(data_path = TRAIN_PATH, batch_size=BATCH_SIZE)
# img = train_data.next_batch()[0]   (16,32,32,1), (16,1), True/False
# plt.imshow(img[0])

# Create training model
train_model = ANVAE(BATCH_SIZE, LEARNING_RATE_AE, LEARNING_RATE_DISC, LEARNING_RATE_GEN)

# Create checkpoint to save training state
checkpoint = tf.train.Checkpoint(model=train_model)

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

            

    

    
