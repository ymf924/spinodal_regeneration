# -*- coding: utf-8 -*-
"""
@File    : tf_ae.py
@Time    : 8/3/2021 3:38 PM
@Author  : Mengfei Yuan
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Lambda, Dense, Input, Reshape, Flatten, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Nadam, Adam, RMSprop
from tensorflow.keras.utils import plot_model


class AutoEncoder():

    def __init__(self, img_size=28, channels=1):
        self.img_size = img_size
        self.channels = channels
        self.original_dim = img_size * img_size

    def build_dense_autoencoder(self,):
        # encoder
        shape = (self.original_dim,)
        input1 = Input(shape=shape, name='encoder_input')
        x = Dense(256, activation='relu')(input1)
        x = Dense(32, activation='relu')(x)
        encoder = Model(inputs=input1, outputs=x, name='encoder')
        encoder.summary()

        # decoder
        latent_inputs = Input(shape=x.shape[1:], name='latent_input')
        x = Dense(256, activation='relu')(latent_inputs)
        x = Dense(self.original_dim, activation='relu')(x)
        decoder = Model(inputs=latent_inputs, outputs=x, name='decoder')
        decoder.summary()

        # AE model
        output1 = decoder(encoder(input1))
        model = Model(inputs=input1, outputs=output1, name='dense_autoencoder')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        # plot_model(encoder, to_file='../save_model/ae_dense_encoder.png', show_shapes=True)
        # plot_model(decoder, to_file='../save_model/ae_dense_decoder.png', show_shapes=True)
        # plot_model(model, to_file='../save_model/ae_dense_model.png', show_shapes=True)
        return model, encoder, decoder

    def build_cnn_autoencoder(self,):
        # encoder
        shape = (self.img_size, self.img_size, self.channels)
        input1 = Input(shape=shape, name='encoder_input')
        x = Conv2D(64, (4,4), strides=[2,2], padding='valid', activation='relu')(input1)
        x = Conv2D(128, (4,4), strides=[2,2], padding='valid', activation='relu')(x)
        x = Conv2D(512, (2,2), strides=[2,2], padding='valid', activation='relu')(x)
        encoder = Model(inputs=input1, outputs=x, name='encoder')
        encoder.summary()

        # decoder
        latent_inputs = Input(shape=x.shape[1:], name='latent_input')
        x = Conv2DTranspose(128, (4,4), strides=[2,2], padding='valid', activation='relu')(latent_inputs)
        x = Conv2DTranspose(64, (4,4), strides=[2,2], padding='valid', activation='relu')(x)
        x = Conv2DTranspose(1, (2,2), strides=[2,2], padding='valid', activation='relu')(x)
        decoder = Model(inputs=latent_inputs, outputs=x, name='decoder')
        decoder.summary()

        # AE model
        output1 = decoder(encoder(input1))
        model = Model(inputs=input1, outputs=output1, name='dense_autoencoder')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(loss='mse', optimizer=opt)
        model.summary()
        # plot_model(encoder, to_file='../save_model/ae_cnn_encoder.png', show_shapes=True)
        # plot_model(decoder, to_file='../save_model/ae_cnn_decoder.png', show_shapes=True)
        # plot_model(model, to_file='../save_model/ae_cnn_model.png', show_shapes=True)
        return model, encoder, decoder

    def build_dense_encoder_cls(self, input_shape=(32,), num_class=10):
        """latent space to classification"""
        input1 = Input(shape=input_shape, name='input1')
        x = Dense(256, activation='relu')(input1)
        x = Dense(num_class + 1, activation='softmax')(x)
        encoder = Model(inputs=input1, outputs=x, name='encoder_cls')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder

    def build_dense_encoder_zmeans(self, input_shape=(32,), latent_dim=2):
        """latent space to classification"""
        input1 = Input(shape=input_shape, name='input1')
        x = Dense(256, activation='relu')(input1)
        x = Dense(latent_dim)(x)  # , activation='relu'
        encoder = Model(inputs=input1, outputs=x, name='encoder_zmeans')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder

    def build_cnn_encoder_cls(self, input_shape=(2,2,512), num_class=10):
        """latent space to classification"""
        input1 = Input(shape=input_shape, name='input1')
        x = Reshape((-1, ))(input1)
        x = Dense(256, activation='relu')(x)
        x = Dense(num_class + 1, activation='softmax')(x)
        encoder = Model(inputs=input1, outputs=x, name='encoder_cls')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder

    def build_cnn_encoder_zmeans(self, input_shape=(2,2,512), latent_dim=2):
        """latent space to classification"""
        input1 = Input(shape=input_shape, name='input1')
        x = Reshape((-1,))(input1)
        x = Dense(256, activation='relu')(x)
        x = Dense(latent_dim)(x)  # , activation='relu'
        encoder = Model(inputs=input1, outputs=x, name='encoder_zmeans')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder


if __name__ == "__main__":
    ae = AutoEncoder(img_size=28, channels=1)
    dense_ae = ae.build_dense_autoencoder()
    cnn_ae = ae.build_cnn_autoencoder()


