# -*- coding: utf-8 -*-
"""
@File    : run_tf_ae.py
@Time    : 8/3/2021 3:39 PM
@Author  : Mengfei Yuan
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau

from tf_ae import AutoEncoder
from label2params import label2param, get_params
from utils import load_data, train_test_split_balanced
from utils import plot_fig_recons, get_classification_report, plot_zmeans, plot_true_prediction


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
    "model_type":'ae',
    "data_type": 'spinodal_28_reg',  # spinodal_28, spinodal_64, mnist
    "nn_type": "cnn",  # dense, cnn
    "img_size": 28,
    "img_channels": 1,
    "batch_size": 64,
    "epochs": 31,

    "data_dir": "../../data/",
    "model_dir": "../save_model_ae/tf/",

    "recon_loss_type": 'bce',  # bce, mse
    "is_bn": False,  # add batch_normalization or not
    "is_load_pretrain": False
}

# adding configs for saving plots and weights
config['img_dir'] = "%s/%s/%s_img_%s_%s_%s_bn%s/" %(
    config['model_dir'], config['data_type'], config["model_type"],
    config['data_type'], config['nn_type'], config['recon_loss_type'], config['is_bn'],)
os.makedirs(config["img_dir"], exist_ok=True)

config["fig_loss"] = "%s/loss.png" % (config["img_dir"])
config["fig_recons"] = "%s/recon_example.png" % (config["img_dir"])
config["fig_zmeans"] = "%s/param_predictions.png" % (config["img_dir"])
config["cls_report"] = "%s/cls_report.txt" % (config["img_dir"])
config["fig_reg_true_prediction"] = "%s/reg_true_prediction" % (config["img_dir"])

config["load_model_weights"] = "%s/%s_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])
config["load_enc_weights"] = "%s/%s_enc_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])
config["load_dec_weights"] = "%s/%s_dec_%s_%s.h5" % (
    config["img_dir"], config["model_type"], config["data_type"], config["nn_type"])

if __name__ == "__main__":
    print('config:\n', config)

    # network type: dense for flatten img, cnn for normal img
    if config["nn_type"] == "dense":
        is_data_flatten = True
    elif config["nn_type"] == "cnn":
        is_data_flatten = False

    # load data
    x, y = load_data(data_dir=config["data_dir"], data_type=config["data_type"], is_extra_normalization=False)
    x_train, x_test, y_train, y_test = train_test_split_balanced(x, y, test_size=0.2, img_size=28, is_flatten=is_data_flatten)

    # build and compile model
    ae = AutoEncoder(img_size=config["img_size"], channels=config["img_channels"])
    if config["nn_type"] == 'dense':
        model, encoder, decoder = ae.build_dense_autoencoder()
    elif config["nn_type"] == 'cnn':
        model, encoder, decoder = ae.build_cnn_autoencoder()

    # train
    if config['is_load_pretrain']:
        model.load_weights(config["load_weights"])
    else:
        es = EarlyStopping(monitor='loss', patience=2, verbose=0, mode='auto')
        rp = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=0, verbose=0, mode='auto',
                               min_delta=0.0001, cooldown=0, min_lr=0)
        history = model.fit(x_train, x_train, epochs=config["epochs"], batch_size=config['batch_size'],
                            validation_data=(x_test, x_test), callbacks=[es, rp])
        model.save_weights(config["load_model_weights"])
        encoder.save_weights(config["load_enc_weights"])
        decoder.save_weights(config["load_dec_weights"])

    # evaluation
    x_test_pred = model.predict(x_test)
    config["L1_norm_recon"] = str(np.mean(np.abs(
        x_test_pred.reshape(-1, config['img_size'] ** 2) - x_test.reshape(-1, config['img_size'] ** 2))))
    plot_fig_recons(x_test, x_test_pred, config)

    # ----------------------------
    # - map to parameters space  -
    # ----------------------------

    x_train_latent = encoder.predict(x_train)
    x_test_latent = encoder.predict(x_test)

    # 3. use trained encoder to do regression for vtk2D_512to32_2dcontinuous_10k.csv
    if config['data_type'].endswith('reg'):
        y_train_param = y_train
        y_test_param = y_test

        if config['nn_type'] == 'dense':
            encoder_zmeans = ae.build_dense_encoder_zmeans(input_shape=(x_train_latent.shape[1:]), latent_dim=2)
        elif config['nn_type'] == 'cnn':
            encoder_zmeans = ae.build_cnn_encoder_zmeans(input_shape=(x_train_latent.shape[1:]), latent_dim=2)

        history = encoder_zmeans.fit(x_test_latent, y_test_param,
                                     epochs=20, batch_size=64,
                                     validation_data=(x_test_latent, y_test_param))  #, callbacks=[es, rp]
        test_loss, test_acc = encoder_zmeans.evaluate(x_test_latent, y_test_param)
        print('encoder_zmeans test accuracy: %s', test_acc)
        zmeans = encoder_zmeans.predict(x_test_latent)
        config["L1_norm_param_predict"] = str(np.mean(np.abs(zmeans - y_test_param)))
        plot_true_prediction(zmeans, y_test, config["fig_reg_true_prediction"])


    # #  1. use trained encoder to do classification
    # num_class = len(np.unique(y_train))
    # if config['nn_type'] == 'dense':
    #     encoder_cls = ae.build_dense_encoder_cls(input_shape=(x_train_latent.shape[1:]), num_class=num_class)
    # elif config['nn_type'] == 'cnn':
    #     encoder_cls = ae.build_cnn_encoder_cls(input_shape=(x_train_latent.shape[1:]), num_class=num_class)
    # history = encoder_cls.fit(x_train_latent, y_train,
    #                           epochs=config["epochs"], batch_size=config['batch_size'],
    #                           validation_data=(x_test_latent, y_test), callbacks=[es, rp])
    # test_loss, test_acc = encoder_cls.evaluate(x_test_latent, y_test)
    # print('encoder_cls test accuracy: %s', test_acc)
    # y_pred = encoder_cls.predict(x_test_latent)
    # y_pred_01, accuracy, report = get_classification_report(y_test, y_pred)
    # with open(config["cls_report"], 'w', encoding='utf-8') as f:
    #     f.write(report)
    #
    # # 2. use trained encoder to map latent space to param space
    # if config['data_type'].startswith('spinodal'):
    #     y_train_param = get_params(y_train, label2param)
    #     y_test_param = get_params(y_test, label2param)
    #
    #     if config['nn_type'] == 'dense':
    #         encoder_zmeans = ae.build_dense_encoder_zmeans(input_shape=(x_train_latent.shape[1:]), latent_dim=2)
    #     elif config['nn_type'] == 'cnn':
    #         encoder_zmeans = ae.build_cnn_encoder_zmeans(input_shape=(x_train_latent.shape[1:]), latent_dim=2)
    #     history = encoder_zmeans.fit(x_test_latent, y_test_param,
    #                                  epochs=config["epochs"], batch_size=config['batch_size'],
    #                                  validation_data=(x_test_latent, y_test_param), callbacks=[es, rp])
    #     test_loss, test_acc = encoder_zmeans.evaluate(x_test_latent, y_test_param)
    #     print('encoder_zmeans test accuracy: %s', test_acc)
    #     zmeans = encoder_zmeans.predict(x_test_latent)
    #     config["L1_norm_param_predict"] = str(np.mean(np.abs(zmeans - y_test_param)))
    #     plot_zmeans(zmeans, y_test, config["fig_zmeans"])
    #
    #
    # # save experimental configs
    # with open(config["img_dir"] + 'config.json', 'w', encoding='utf-8') as f:
    #     json.dump(config, f, indent=4, ensure_ascii=False)
    # print('save all to %s', config['img_dir'])


