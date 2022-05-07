import tensorflow as tf
import utils
# arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
#         arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
#         arch_cells['normal_dec'] = ['mconv_e6k5g0']
#         arch_cells['up_dec'] = ['mconv_e6k5g0']
#         arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
#         arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
#         arch_cells['normal_post'] = ['mconv_e3k5g0']
#         arch_cells['up_post'] = ['mconv_e3k5g0']
#         arch_cells['ar_nn'] = ['']

BN_EPS = 1e-5
CHANNEL_MULT = 2

class Encoder(tf.keras.Model):
    def __init__(self, mult, latent_layers, nodes_per_layer, cells_per_node, enc_num_c, dec_num_c):
        super(Encoder, self).__init__()
        self.encoder_cells, self.mult = self.init_encoder_cells(mult, latent_layers, nodes_per_layer, cells_per_node, enc_num_c, dec_num_c)

    def init_encoder_cells(self, mult, latent_layers, nodes_per_layer, cells_per_node, enc_num_c, dec_num_c):
        with tf.name_scope("encoder"):
            encoder_cells = []
            for l in range(latent_layers):
                for n in range(nodes_per_layer):
                    for e in range(cells_per_node):    
                        num_c = enc_num_c * mult
                        cell = Cell(num_c, num_c, "normalEncoderCell")
                        encoder_cells.append(cell)

                    if not (l == latent_layers - 1 and n == nodes_per_layer - 1):
                        num_c_enc = enc_num_c*mult
                        num_c_dec = dec_num_c*mult
                        cell = combinerEncoderCell(num_c_enc, num_c_dec, num_c_enc)
                        encoder_cells.append(cell)
                
                if l < latent_layers - 1:
                    num_ci = enc_num_c*mult
                    num_co = CHANNEL_MULT * num_ci
                    cell = Cell(num_ci, num_co, "downSampleEncoderCell")
                    encoder_cells.append(cell)
                    mult = CHANNEL_MULT*mult
        
        return encoder_cells, mult

    def call(self):
        raise NotImplementedError

class Decoder(tf.keras.Model):
    def __init__(self, mult, latent_layers, nodes_per_layer, cells_per_node, latent_channel_dim, enc_num_c, dec_num_c):
        super(Decoder, self).__init__()
        self.decoder_cells, self.mult = self.init_decoder_cells(mult, latent_layers, nodes_per_layer, cells_per_node, latent_channel_dim, enc_num_c, dec_num_c)

    def init_decoder_cells(self, mult, latent_layers, nodes_per_layer, cells_per_node, latent_channel_dim, enc_num_c, dec_num_c):
        with tf.name_scope("decoder"):
            decoder_cells = []
            for l in range(latent_layers):
                for n in range(nodes_per_layer):
                    num_c = dec_num_c * mult 
                    if not (l == 0 and n == 0):
                        for e in range(cells_per_node):
                            cell = Cell(num_c, num_c, "normalDecoderCell")
                            decoder_cells.append(cell)

                    cell = combinerDecoderCell(num_c, latent_channel_dim, num_c)
                    decoder_cells.append(cell)

                if l < latent_layers - 1:
                    num_ci = dec_num_c * mult
                    num_co = num_ci / CHANNEL_MULT
                    cell = Cell(num_ci, num_co, "upSampleDecoderCell")
                    decoder_cells.append(cell)
                    mult = mult/CHANNEL_MULT

        return decoder_cells, mult

    def call(self):
        raise NotImplementedError

class Discriminator(tf.keras.Model):
    def __init__(self, latent_layers):
        super(Discriminator, self).__init__()
        self.discriminator_cells = self.init_discriminator_cells(latent_layers)
        
    def init_discriminator_cells(self, latent_layers):
        with tf.name_scope("discriminator"):
            discriminator_cells = []
            for l in range(latent_layers):
                cell = DiscriminatorCell()
                discriminator_cells.append(cell)

        return discriminator_cells

class ImageConditional(tf.keras.Model):
    def __init__(self, dec_num_c, mult):
        super(ImageConditional, self).__init__()
        self.image_conditional = self.init_image_conditional(dec_num_c, mult)

    def init_image_conditional(self, dec_num_c, mult):
        cin = dec_num_c * mult
        cout = 1
        return tf.keras.Sequential([
            tf.keras.layers.ELU(),
            tf.keras.layers.Conv2D(cout, kernel_size=3, padding = "same", use_bias=True)
        ])

class Cell(tf.keras.Model):
    def __init__(self, cin, cout, cell_type):
        super(Cell, self).__init__()
        self.cell_type = cell_type
        strides = 1
        if cell_type.startswith('normal') or cell_type.startswith('combiner'):
            strides = 1
        elif cell_type.startswith('down'):
            strides = 2
        elif cell_type.startswith('up'):
            strides = -1
        self.skip = utils.get_skip_connection(cin, strides, mult=CHANNEL_MULT)
        op_list = {
            "normalEncoderCell": normalEncoderCell(cin, cout),
            "downSampleEncoderCell": downSampleEncoderCell(cin, cout),
            "normalDecoderCell": normalDecoderCell(cin, cout),
            "upSampleDecoderCell": upSampleDecoderCell(cin, cout)
        }
        self.op = op_list[cell_type]

    def call(self, x):
        skip = self.skip(x)
        x = self.op(x)
        return skip + 0.1*x

class normalEncoderCell(tf.keras.Model):
    def __init__(self, cin, cout, kernel=3, strides=1, padding="same", dilation=1):
        super(normalEncoderCell, self).__init__()
        # Paper uses third-party implementation for bnswish -> Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        self.seq = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(cout, kernel_size=kernel, strides=strides, padding=padding, use_bias=True, name="enc_norm_Conv2D")
        ])
        
    def call(self, x):
        return self.seq(x)

class combinerEncoderCell(tf.keras.Model):
    def __init__(self, cin1, cin2, cout):
        super(combinerEncoderCell, self).__init__()
        self.seq = tf.keras.layers.Conv2D(cout, kernel_size=1, strides=1, padding="same", use_bias=True, name="enc_comb_Conv2D")

    def call(self, x1, x2):
        x2 = self.seq(x2)
        return x1+x2

class downSampleEncoderCell(tf.keras.Model):
    def __init__(self, cin, cout, kernel = 3, strides=2, padding="same", dilation=1):
        super(downSampleEncoderCell, self).__init__()
        # Paper uses third-party implementation for bnswish -> Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        self.seq = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Activation('swish'),
            tf.keras.layers.Conv2D(cout, kernel_size=kernel, strides=strides, padding=padding, use_bias=True, name="enc_down_Conv2D")
        ])

    def call(self, x):
        return self.seq(x)

class normalDecoderCell(tf.keras.Model):
    def __init__(self, cin, cout, strides=1, ex=6, dil=1, k=5, g=0):
        super(normalDecoderCell, self).__init__()
        hidden_dim = int(round(cin*ex))
        groups = hidden_dim if g == 0 else g

        self.seq = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Conv2D(hidden_dim, kernel_size=1, strides=1, padding="same", groups=1, dilation_rate=1, use_bias=False, name="dec_norm_Conv2D_1"),
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Conv2D(hidden_dim, kernel_size=k, strides=strides, padding="same", groups=groups, dilation_rate=dil, use_bias=False, name="dec_norm_Conv2D_2"),
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Conv2D(cout, kernel_size=1, strides=1, padding="same", use_bias=False, name="dec_norm_Conv2D_3"),
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS)
        ])

    def call(self, x):
        return self.seq(x)

class combinerDecoderCell(tf.keras.Model):
    def __init__(self, cin1, cin2, cout):
        super(combinerDecoderCell, self).__init__()
        self.seq = tf.keras.layers.Conv2D(cout, kernel_size=1, strides=1, padding="same", use_bias=True, name="dec_comb_Conv2D")

    def call(self, x1, x2):
        out = tf.concat([x1, x2], -1)
        return self.seq(out)

class upSampleDecoderCell(tf.keras.Model):
    def __init__(self, cin, cout, strides=1, ex=6, dil=1, k=3, g=0):
        super(upSampleDecoderCell, self).__init__()
        hidden_dim = int(round(cin*ex))
        groups = hidden_dim if g == 0 else g

        self.seq = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(size=2),
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Conv2D(hidden_dim, kernel_size=1, strides=1, padding="same", groups=1, dilation_rate=1, use_bias=False, name="dec_up_Conv2D_1"),
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Conv2D(hidden_dim, kernel_size=k, strides=strides, padding="same", groups=groups, dilation_rate=dil, use_bias=False, name="dec_up_Conv2D_2"),
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS),
            tf.keras.layers.Conv2D(cout, kernel_size=1, strides=1, padding="same", use_bias=False, name="dec_up_Conv2D_3"),
            tf.keras.layers.BatchNormalization(momentum=0.05, epsilon=BN_EPS)
        ])

    def call(self, x):
        return self.seq(x)

class DiscriminatorCell(tf.keras.Model):
    def __init__(self, h_dim = 1000):
        super(DiscriminatorCell, self).__init__()
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(h_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(h_dim),
                tf.keras.layers.LeakyReLU(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1)
            ]
        )
        
    def call(self, x):
        return self.seq(x)