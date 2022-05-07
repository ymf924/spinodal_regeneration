import tensorflow as tf
import matplotlib.pyplot as plt

class FactorizedReduce(tf.keras.Model):
    def __init__(self, cin, cout):
        super(FactorizedReduce, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(cout // 4, kernel_size = 1, strides = 2, padding = "same", use_bias = True)
        self.conv_2 = tf.keras.layers.Conv2D(cout // 4, kernel_size = 1, strides = 2, padding = "same", use_bias = True)
        self.conv_3 = tf.keras.layers.Conv2D(cout // 4, kernel_size = 1, strides = 2, padding = "same", use_bias = True)
        self.conv_4 = tf.keras.layers.Conv2D(cout-3 * (cout // 4), kernel_size = 1, strides = 2, padding = "same", use_bias = True)

    def call(self, x):
        out = tf.keras.layers.Activation('swish')(x)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:, 1:, 1:, :])
        conv3 = self.conv_3(out[:, :, 1:, :])
        conv4 = self.conv_4(out[:, 1:, :, :])
        out = tf.concat([conv1, conv2, conv3, conv4], -1)
        return out

def get_skip_connection(C, strides, mult):
    if strides == 1:
        return Identity()
    elif strides == 2:
        return FactorizedReduce(C, int(mult*C))
    elif strides == -1:
        return UpSample(C, mult)

class UpSample(tf.keras.Model):
    def __init__(self, C, mult):
        super(UpSample, self).__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(2, interpolation="bilinear"),
            tf.keras.layers.Conv2D(int(C/mult), kernel_size=1)
        ])

    def call(self, x):
        return self.seq(x)

class Identity(tf.keras.Model):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x):
        return x

def norm(x, axis):
        return tf.math.sqrt(tf.math.reduce_sum(x*x, axis=axis))
