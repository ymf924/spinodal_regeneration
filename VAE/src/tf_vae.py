# -*- coding: utf-8 -*-
"""
@File    : tf_vae.py
@Time    : 8/4/2021 3:26 PM
@Author  : Mengfei Yuan
"""
# add bn layer to latent dimension
# https://spaces.ac.cn/archives/7381/comment-page-1

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer, Lambda, Dense, Input, Reshape, Flatten, Activation
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Nadam, Adam, RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.losses import mse, binary_crossentropy


class VAE():

    def __init__(self, img_size=28, channels=1, config={}):
        self.img_size = img_size
        self.channels = channels
        self.original_dim = img_size * img_size
        self.recon_loss_type = config["recon_loss_type"]
        self.is_bn = config["is_bn"]
        self.z_dim = config["z_dim"]

    def build_dense_vae(self):
        input_shape = (self.original_dim,)
        latent_dim = self.z_dim # 2

        # encoder
        input1 = Input(shape=input_shape, name='encoder_input')
        x = Dense(256, activation='relu')(input1)
        x = Dense(32, activation='relu')(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_std = Dense(latent_dim, name='z_std')(x)
        if self.is_bn:
            z_mean, z_std = self.add_bn(z_mean, z_std)
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_std])
        encoder = Model(inputs=input1, outputs=[z_mean, z_std, z], name='encoder')
        encoder.summary()

        # decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(32, activation='relu')(latent_inputs)
        x = Dense(256, activation='relu')(x)
        x = Dense(self.original_dim, activation='sigmoid')(x)
        decoder = Model(inputs=latent_inputs, outputs=x, name='decoder')
        decoder.summary()

        # vae
        output1 = decoder(encoder(input1)[-1])
        model = Model(inputs=input1, outputs=output1, name='dense_vae')
        # add vae loss = recon_loss + kl_loss
        recon_loss, kl_loss, vae_loss = self.vae_loss_func(input1, output1, z_mean, z_std)
        model.add_loss(vae_loss)
        model.add_metric(recon_loss, name='recon_loss', aggregation='mean')
        model.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        model.add_metric(vae_loss, name='vae_loss', aggregation='mean')
        # opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(optimizer=Adam(1e-4))
        model.summary()
        # plot_model(encoder, to_file='../save_model/vae_dense_encoder.png', show_shapes=True)
        # plot_model(decoder, to_file='../save_model/vae_dense_decoder.png', show_shapes=True)
        # plot_model(model, to_file='../save_model/vae_dense_model.png', show_shapes=True)
        return model, encoder, decoder

    def build_cnn_vae(self):
        input_shape = (self.img_size, self.img_size, self.channels)
        latent_dim = self.z_dim # 2

        # encoder
        input1 = Input(shape=input_shape, name='encoder_input')
        x = Conv2D(64, (4, 4), strides=[2, 2], padding='valid', activation='relu')(input1)
        x = Conv2D(128, (4, 4), strides=[2, 2], padding='valid', activation='relu')(x)
        x = Conv2D(512, (2, 2), strides=[2, 2], padding='valid', activation='relu')(x)
        x = Reshape((-1, ))(x)
        z_mean = Dense(latent_dim, name='z_mean')(x)
        z_std = Dense(latent_dim, name='z_std')(x)
        if self.is_bn:
            z_mean, z_std = self.add_bn(z_mean, z_std)
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_std])
        encoder = Model(inputs=input1, outputs=[z_mean, z_std, z], name='encoder')
        encoder.summary()

        # decoder
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(2*2*512)(latent_inputs)
        x = Reshape((2, 2, 512))(x)
        x = Conv2DTranspose(128, (4, 4), strides=[2, 2], padding='valid', activation='relu')(x)
        x = Conv2DTranspose(64, (4, 4), strides=[2, 2], padding='valid', activation='relu')(x)
        x = Conv2DTranspose(1, (2, 2), strides=[2, 2], padding='valid', activation='sigmoid')(x)
        decoder = Model(inputs=latent_inputs, outputs=x, name='decoder')
        decoder.summary()

        # vae
        output1 = decoder(encoder(input1)[-1])
        model = Model(inputs=input1, outputs=output1, name='cnn_vae')
        # add vae loss = recon_loss + kl_loss
        recon_loss, kl_loss, vae_loss = self.vae_loss_func(input1, output1, z_mean, z_std)
        model.add_loss(vae_loss)
        model.add_metric(recon_loss, name='recon_loss', aggregation='mean')
        model.add_metric(kl_loss, name='kl_loss', aggregation='mean')
        model.add_metric(vae_loss, name='vae_loss', aggregation='mean')
        # opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        model.compile(optimizer=Adam(1e-4))
        model.summary()
        # plot_model(encoder, to_file='../save_model/vae_cnn_encoder.png', show_shapes=True)
        # plot_model(decoder, to_file='../save_model/vae_cnn_decoder.png', show_shapes=True)
        # plot_model(model, to_file='../save_model/vae_cnn_model.png', show_shapes=True)

        return model, encoder, decoder

    def build_dense_encoder_cls(self, input_shape=(64,), num_class=10):
        """latent space to classification"""
        input1 = Input(shape=input_shape, name='input1')
        x = Dense(256, activation='relu')(input1)
        x = Dense(num_class + 1, activation='softmax')(x)
        encoder = Model(inputs=input1, outputs=x, name='encoder_cls')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder

    def build_dense_encoder_zmeans(self, input_shape=(64,), latent_dim=2):
        """latent space to classification"""
        input1 = Input(shape=input_shape, name='input1')
        x = Dense(256, activation='relu')(input1)
        x = Dense(latent_dim)(x)  # , activation='relu'
        encoder = Model(inputs=input1, outputs=x, name='encoder_zmeans')
        opt = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder

    def vae_loss_func(self, y_true, y_pred, z_mean, z_std):
        if self.recon_loss_type == 'mse':
            recon_loss = mse(y_true, y_pred)
        elif self.recon_loss_type == 'bce':
            recon_loss = binary_crossentropy(y_true, y_pred)
        recon_loss *= self.original_dim
        recon_loss = K.mean(recon_loss)  # have to add this for conv otherwise back gradient dimension will be mismatched

        kl_loss = 1 + z_std - K.square(z_mean) - K.exp(z_std)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(recon_loss + kl_loss)
        return recon_loss, kl_loss, vae_loss

    def add_bn(self, z_mean, z_std):
        # add batch normalization to keep KL vanishing away
        scaler = Scaler()
        z_mean = BatchNormalization(scale=False, center=False, epsilon=1e-8)(z_mean)
        z_mean = scaler(z_mean, mode='positive')
        z_std = BatchNormalization(scale=False, center=False, epsilon=1e-8)(z_std)
        z_std = scaler(z_std, mode='positive')
        return z_mean, z_std


class Scaler(Layer):
    """特殊的scale层
    """

    def __init__(self, tau=0.5, **kwargs):
        super(Scaler, self).__init__(**kwargs)
        self.tau = tau

    def build(self, input_shape):
        super(Scaler, self).build(input_shape)
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],), initializer='zeros'
        )

    def call(self, inputs, mode='positive'):
        if mode == 'positive':
            scale = self.tau + (1 - self.tau) * K.sigmoid(self.scale)
        else:
            scale = (1 - self.tau) * K.sigmoid(-self.scale)
        return inputs * K.sqrt(scale)

    def get_config(self):
        config = {'tau': self.tau}
        base_config = super(Scaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def sampling(inputs):
    """重参数采样
    """
    z_mean, z_std = inputs
    noise = K.random_normal(shape=K.shape(z_mean))
    # z_std (always positive require activation function) or z_std_log (does not require activation function)
    return z_mean + K.exp(0.5 * z_std) * noise
    # return z_mean + z_std * noise


if __name__ == "__main__":
    config = {
    "model_type": "vae",
    "data_type": 'spinodal_28',  # spinodal_28, spinodal_64, mnist
    "nn_type": "dense",  # dense, cnn
    "img_size": 28,
    "img_channels": 1,
    "batch_size": 64,
    "epochs": 11,

    "data_dir": "../../data/",
    "model_dir": "../save_model/tf/",

    "recon_loss_type": 'bce',  # bce, mse
    "is_bn": False,  # add batch_normalization or not
    "is_load_pretrain": False,

    "z_dim": 64,
}

    vae = VAE(img_size=28, channels=1, config=config)
    dense_vae = vae.build_dense_vae()
    cnn_vae = vae.build_cnn_vae()
