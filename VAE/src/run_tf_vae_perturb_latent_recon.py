# -*- coding: utf-8 -*-
"""
@File    : run_tf_vae_perturb_latent_recon.py
@Time    : 11/18/2021 7:57 PM
@Author  : Mengfei Yuan
"""



import os
import random
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from tf_vae import VAE
from label2params import label2param, get_params
from utils import load_data, train_test_split_balanced
from utils import plot_one_sample
import matplotlib.pyplot as plt



# setting tf gpu usage
gpus = tf.config.experimental.list_physical_devices('GPU')
print('gpus: ', gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# configs

config = {
    "model_type": "vae",
    "data_type": 'spinodal_28_reg',  # spinodal_28, spinodal_64, mnist
    "nn_type": "cnn",  # dense, cnn
    "img_size": 28,
    "img_channels": 1,
    "batch_size": 64,
    "epochs": 81,

    "data_dir": "../../data/",
    "model_dir": "../save_model_vae/tf/",

    "recon_loss_type": 'bce',  # bce, mse
    "is_bn": False,  # add batch_normalization or not
    "is_load_pretrain": True,

    "z_dim": 32,
}

# type and value for perturbation
config["perturbation_mode"] = "gaussian"  # gaussian, uniform
perturb_std_rate = 0.05  # random std added on perturbation
perturb_mu_rate = 0.1  # +- perturbation mean added on latent space
num_fake = 16  # how many perturbed recon are displayed

# adding configs for saving plots and weights
config['img_dir'] = "%s/%s/%s_img_%s_%s_%s_bn%s/" %(
    config['model_dir'], config['data_type'], config["model_type"],
    config['data_type'], config['nn_type'], config['recon_loss_type'], config['is_bn'],)
os.makedirs(config["img_dir"], exist_ok=True)

config["load_model_weights"] = "%s/%s_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])
config["load_enc_weights"] = "%s/%s_enc_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])
config["load_dec_weights"] = "%s/%s_dec_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])

if __name__ == "__main__":
    print("configs: ", config)
    
    # create a folder to store all the results
    config["latent_perturb_dir"] = "../latent_perturb_dir/"
    if not os.path.exists(config["latent_perturb_dir"]):
        os.mkdir(config["latent_perturb_dir"])
        
    # network type: dense for flatten img, cnn for normal img
    if config["nn_type"] == "dense":
        is_data_flatten = True
    elif config["nn_type"] == "cnn":
        is_data_flatten = False
        
    # build and compile vae model
    vae = VAE(img_size=config["img_size"], channels=config["img_channels"], config=config)
    if config["nn_type"] == 'dense':
        model, encoder, decoder = vae.build_dense_vae()
    elif config["nn_type"] == 'cnn':
        model, encoder, decoder = vae.build_cnn_vae()
     
    # load model, encoder, decoder weigths
    if config['is_load_pretrain']:
        model.load_weights(config["load_model_weights"])
        encoder.load_weights(config["load_enc_weights"])
        decoder.load_weights(config["load_dec_weights"])


        
    # load data, load 6 class ic and kappa data
    x, y = load_data(data_dir=config["data_dir"], data_type="spinodal_28", is_extra_normalization=False)
    x_train, x_test, y_train, y_test = train_test_split_balanced(x, y, test_size=0.2, img_size=28, is_flatten=is_data_flatten)


    #-------------------------------------------------------
    # perturbed sample
    #-------------------------------------------------------

    # select one sample from each class to generate perturbed recon
    # label2param = {
    #     1: [0, 0.25],
    #     2: [0, 0.7],
    #     3: [0.1, 0.25],
    #     4: [0.1, 0.7],
    #     5: [0.2, 0.25],
    #     6: [0.2, 0.7],
    # }

    x1, y1 = [], []
    for c in np.unique(y_train):
        for xi, yi in zip(x_train, y_train):
            if yi == c:
                x1.append(xi)
                y1.append(yi)
                break

    for x_fixed, y_fixed in zip(x1, y1):
        print('-------------------------')
        print('class ', str(int(y_fixed)))
        print('ic=%s, kappa=%s' %(str(label2param[y_fixed][0]), str(label2param[y_fixed][1])))

        # plot true spinodal image
        if is_data_flatten == True:
            x_fixed_plot = x_fixed.reshape(28, 28)
        else:
            x_fixed_plot = x_fixed

        print(config["latent_perturb_dir"]+'true_class_'+str(int(int(y_fixed)))+'.png')
        plot_one_sample(x_fixed_plot.squeeze(), y_fixed,
                        save_path=config["latent_perturb_dir"]+'class_'+str(int(int(y_fixed)))+'_true.png')

        # get latent from x_fixed
        x_fixed_latent_mean, x_fixed_latent_std, x_fixed_latent = encoder.predict(x_fixed.reshape(1,28,28,1))
        print("vae latent dimension")
        print(x_fixed_latent_mean.shape, x_fixed_latent_std.shape, x_fixed_latent.shape)
        print("latent dimension min and max")
        print(x_fixed_latent.max(), x_fixed_latent.min())
        x_fixed_latent_plot = x_fixed_latent.reshape(8, int(config["z_dim"]/8))
        plot_one_sample(x_fixed_latent_plot, y_fixed,
                        save_path=config["latent_perturb_dir"]+'class_'+str(int(int(y_fixed)))+'_true_latent.png')

        # recon from latent (no perturbation)
        x_fixed_recon = decoder.predict(x_fixed_latent)
        x_fixed_recon_plot = x_fixed_recon.reshape(28, 28)
        plot_one_sample(x_fixed_recon_plot, y_fixed,
                        save_path=config["latent_perturb_dir"]+'class_'+str(int(int(y_fixed)))+'_recon.png')

        # recon from latent (with perturbation)
        # 1. set perturbation
        perturb_std = (x_fixed_latent.max() - x_fixed_latent.min()) * perturb_std_rate
        perturb_mu = (x_fixed_latent.max() - x_fixed_latent.min()) * perturb_mu_rate

        mu = np.linspace(-perturb_mu, perturb_mu, num_fake)
        # different type of std
        if config["perturbation_mode"] == "uniform":
            std = [np.array([random.uniform(-perturb_std, perturb_std) + mu_i for _ in range(
                config['z_dim'])]).reshape(1, config["z_dim"]) for mu_i in mu]
        elif config["perturbation_mode"] == "gaussian":
            std = [np.array([random.gauss(mu_i, perturb_std) for _ in range(
                config['z_dim'])]).reshape(1, config["z_dim"]) for mu_i in mu]
        x_fixed_latent_perturb = np.array([x_fixed_latent + i for i in std]).squeeze()

        # 2. recon from latent + perturbation
        x_fixed_recon_perturb = decoder.predict(x_fixed_latent_perturb)
        # create multiply image to show the true, recon and recons with perturbation
        # for example: create 5 row 4 columns
        # (5,1) and (5,2) are true and recon, the rest are num_fake recon with perturb
        plt.figure(figsize=(18, 25))
        plt.suptitle('class %s: true, recon and perturbed recon\nic=%s, kappa%s' %(
            str(int(int(y_fixed))), str(label2param[y_fixed][0]), str(label2param[y_fixed][1])))

        ax = plt.subplot(5, 4, 1)
        plt.imshow(x_fixed.squeeze(), cmap=plt.cm.Greys)
        ic = x_fixed.sum() / (28 * 28)
        plt.colorbar()
        ax.set_title('true sample: ' + str(round(ic, 2)))

        ax = plt.subplot(5, 4, 2)
        plt.imshow(x_fixed_recon.squeeze(), cmap=plt.cm.Greys)
        ic = x_fixed_recon.sum() / (28 * 28)
        plt.colorbar()
        ax.set_title('recon sample: ' + str(round(ic, 2)))

        for i in range(0, num_fake):
            ax = plt.subplot(5, 4, i+1+4)
            plt.imshow(x_fixed_recon_perturb[i].squeeze(), cmap=plt.cm.Greys)
            ic = x_fixed_recon_perturb[i].sum() / (28 * 28)
            plt.colorbar()
            ax.set_title('perturb recon %s: %s\nmu=%s, std=%s ' %(
                str(i+1), str(round(ic, 2)), str(round(mu[i], 2)), str(round(perturb_std_rate, 2))))

        plt.savefig(config["latent_perturb_dir"] +
                    "class_"+str(int(y_fixed)) + '_perturb_' +
                    config["perturbation_mode"] + '.png')










