import tensorflow as tf
import tensorflow_probability as tfp
import numpy
from cells import Encoder, Decoder, Discriminator, ImageConditional, Cell, combinerEncoderCell, combinerDecoderCell, DiscriminatorCell
from utils import norm

CHANNEL_MULT = 2

class ANVAE(tf.keras.Model):
    def __init__(self, batch_size, ae_lr, disc_lr, gen_lr):
        super(ANVAE, self).__init__()
        self.batch_size = batch_size
        self.latent_layers = 3
        self.nodes_per_layer = 1
        self.cells_per_node = 1
        self.latent_channel_dim = 10
        self.enc_num_c = 32
        self.dec_num_c = 32
        self.stem = self.init_stem()
        self.input_size = 32

        # Set up encoder cells
        self.encoder = Encoder(1, self.latent_layers, self.nodes_per_layer, self.cells_per_node, self.enc_num_c, self.dec_num_c)
        mult = self.encoder.mult

        # Get initial encoder cell
        self.enc_cell0 = self.encoder_cell0(mult)

        # scaling for sampling
        spatial_scaling = 2 ** (self.latent_layers - 1)
        self.spatial_scaling = spatial_scaling

        c_scaling = CHANNEL_MULT ** (self.latent_layers - 1)
        prior_ftr0_size = (self.batch_size, self.input_size // spatial_scaling, self.input_size // spatial_scaling, int(c_scaling * self.dec_num_c))
        
        # Create random prior feature -> this is tuneable h paramter in NVAE paper 
        self.prior_ftr0 = tf.random.normal(prior_ftr0_size)

        # Set up decoder cells
        self.decoder = Decoder(mult, self.latent_layers, self.nodes_per_layer, self.cells_per_node, self.latent_channel_dim, self.enc_num_c, self.dec_num_c)
        mult = self.decoder.mult

        # Get encoder and decoder sampler cells
        self.enc_sampler, self.dec_sampler = self.init_sampler_cells(mult)

        # Get discriminator cells
        self.discriminator = Discriminator(self.latent_layers)

        self.image_conditional = ImageConditional(self.dec_num_c, mult)
        # self.image_conditional = self.init_image_conditional(mult)

        # Optimizers for training autoencoder, discriminator, and generator
        self.ae_opt = tf.keras.optimizers.Adamax(ae_lr)
        self.gen_opt = tf.keras.optimizers.Adamax(gen_lr)
        self.dc_opt = tf.keras.optimizers.Adamax(disc_lr)
        
        # Next 12 lines are for spectral regularization -> helps stabilize training
        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

        self.all_conv_layers = []
        self.all_bn_layers = []

        for layer in self.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                self.all_conv_layers.append(layer)
            if (isinstance(layer, tf.keras.layers.BatchNormalization)):
                self.all_bn_layers.append(layer)

        self.step_count = 0

        self.log_writer = tf.summary.create_file_writer(logdir='./tf_summary/new')

    def init_stem(self):
        cout = self.enc_num_c
        cin = 1
        stem = tf.keras.layers.Conv2D(cout, kernel_size=3, padding="same", use_bias=True)
        return stem

    def encoder_cell0(self, mult):
        with tf.name_scope("encoder"):
            num_c = self.enc_num_c*mult
            cell = tf.keras.Sequential([
                tf.keras.layers.ELU(),
                tf.keras.layers.Conv2D(2*self.latent_channel_dim, kernel_size=1, padding="same", use_bias=True),
                tf.keras.layers.ELU()
            ])
            return cell

    def init_sampler_cells(self, mult):
        with tf.name_scope("sampler"):
            enc_sampler, dec_sampler = [],[]
            for l in range(self.latent_layers):
                for n in range(self.nodes_per_layer):
                    num_c = self.enc_num_c * mult
                    cell = tf.keras.layers.Conv2D(2*self.latent_channel_dim, kernel_size=3, padding="same", use_bias=True)
                    enc_sampler.append(cell)

                    # add NF later

                    if not(l == 0 and n == 0): 
                        num_c = self.dec_num_c*mult 
                        cell = tf.keras.Sequential([
                            tf.keras.layers.ELU(),
                            tf.keras.layers.Conv2D(2*self.latent_channel_dim, kernel_size=1, padding="same", use_bias=True)
                        ])
                        dec_sampler.append(cell)

                mult = mult/CHANNEL_MULT

            return enc_sampler, dec_sampler

    def init_image_conditional(self, mult):
        cin = self.dec_num_c * mult
        cout = 1
        return tf.keras.Sequential([
            tf.keras.layers.ELU(),
            tf.keras.layers.Conv2D(cout, kernel_size=3, padding = "same", use_bias=True)
        ])

    def call(self, x):
        with tf.GradientTape() as ae_tape, tf.GradientTape() as dc_tape, tf.GradientTape() as gen_tape:

            # input pre processing here
            s = self.stem(2 * x - 1.0)
            print('\n\n s shape: ', s.shape)

            # passing through encoder and saving combiner cells
            combiner_cells_enc = []
            combiner_cells_s = []
            for cell in self.encoder.encoder_cells:
                print('encoder cell name', cell.name)
                print('isinstance:', isinstance(cell, combinerEncoderCell))
                if (isinstance(cell, combinerEncoderCell)):
                    combiner_cells_enc.append(cell)
                    combiner_cells_s.append(s)
                else:
                    s = cell(s)
                print('\n\n looping encoder s shape: ', s.shape)

            combiner_cells_enc.reverse()
            combiner_cells_s.reverse()

            index = 0

            # Create last latent space
            print('\n\n\n Create last latent space')
            ftr = self.enc_cell0(s)
            sample = self.enc_sampler[index](ftr) # sample of last latent space
            mu_q, log_sig_q = tf.split(sample, 2, -1)
            sig_q = tf.math.exp(log_sig_q)

            # Form first prior
            dist = tfp.distributions.Normal(mu_q, log_sig_q)
            z = dist.sample()
            log_q_conv = dist.log_prob(z)
            all_q = [dist]
            all_log_q = [log_q_conv]

            # make sure no deterministic features are passed, reset value of s
            print('\n\n reset value of s s shape: ', s.shape)
            s = tf.random.normal(tf.shape(s))

            # prior for z0 
            dist = tfp.distributions.Normal(tf.zeros_like(z), tf.ones_like(z))
            log_p_conv = dist.log_prob(z)
            all_p = [dist]
            all_log_p = [log_p_conv]

            index = 0

            for cell in self.decoder.decoder_cells:
                print('\n\n decoder cell name', cell.name)
                print('isinstance:', isinstance(cell, combinerDecoderCell))
                if ("combiner" in cell.name):
                    if index > 0:
                        # get prior parameters
                        sample = self.dec_sampler[index-1](s)
                        mu_p, log_sig_p = tf.split(sample, 2, -1)
                        sig_p = tf.math.exp(log_sig_p)

                        # Get feature
                        ftr = combiner_cells_enc[index-1](combiner_cells_s[index-1], s)
                        
                        # get posterior parameters
                        sample = self.enc_sampler[index](ftr)
                        mu_q, log_sig_q = tf.split(sample, 2, -1)
                        sig_q = tf.math.exp(log_sig_q)
                        
                        # Form latent distribution
                        dist = tfp.distributions.Normal(mu_p + mu_q, log_sig_p+log_sig_q)
                        z = dist.sample()
                        
                        log_q_conv = dist.log_prob(z)

                        # add NF later

                        all_log_q.append(log_q_conv)
                        all_q.append(dist)

                        # Form prior
                        dist = tfp.distributions.Normal(mu_p, sig_p)
                        log_p_conv = dist.log_prob(z)
                        all_p.append(dist)
                        all_log_p.append(log_p_conv)

                    # Combiner cell -> this is for relativbe normal distributions, described in NVAE paper
                    s = cell(s, z)
                    index += 1
                else:
                    # Pass through other non-combiner cells normally
                    s = cell(s)

            logits = self.image_conditional.image_conditional(s)
            # logits = self.image_conditional(s)

            self.step_count += 1

            # Get autoencoder loss
            ae_loss = self.autoencoder_loss(x, logits)

            real_samples = []
            fake_samples = []
                
            # Create normal samples for discriminator
            for feature in all_q:
                feat_sample = feature.sample()
                real_samples.append(tf.random.normal(shape=tf.shape(feat_sample), mean=0.0, stddev=1.0))
                fake_samples.append(feat_sample)

            # Get discriminator output
            dc_real = self.discriminator_output(real_samples)
            dc_fake = self.discriminator_output(fake_samples)

            # Discriminator loss
            dc_losses = self.discriminator_loss(dc_real, dc_fake)
            dc_loss = tf.reduce_mean(dc_losses)

            # Generator loss
            gen_loss = self.generator_loss(dc_fake)

            # Get spectral loss
            spectral_loss = self.spectral_norm()

            # Get grads for autoencoder -> encoder + decoder + image processing cells
            ae_grads = ae_tape.gradient(ae_loss + spectral_loss, self.encoder.trainable_variables + self.decoder.trainable_variables + self.image_conditional.trainable_variables)
            # ae_grads = ae_tape.gradient(ae_loss + spectral_loss, self.encoder.trainable_variables + self.decoder.trainable_variables)

            # Get grads for discriminator -> mean of all latent layers
            dc_grads = dc_tape.gradient(dc_loss, self.discriminator.trainable_variables)
            
            # Get generator grads -> encoder
            gen_grads = gen_tape.gradient(gen_loss + spectral_loss, self.encoder.trainable_variables)

        # Apply gradients
        self.ae_opt.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables + self.image_conditional.trainable_variables))
        # self.ae_opt.apply_gradients(zip(ae_grads, self.encoder.trainable_variables + self.decoder.trainable_variables))
        self.dc_opt.apply_gradients(zip(dc_grads, self.discriminator.trainable_variables))
        self.gen_opt.apply_gradients(zip(gen_grads, self.encoder.trainable_variables))

        # Save variables to use in tensorboard
        with self.log_writer.as_default():
            tf.summary.scalar(name='Autoencoder_loss', data=ae_loss, step=self.step_count)
            for i, dc_loss in enumerate(dc_losses):
                tf.summary.scalar(name='Discriminator_Loss_{}'.format(i), data=dc_loss, step=self.step_count)
            tf.summary.scalar(name='Generator_Loss', data=gen_loss, step=self.step_count)
            tf.summary.scalar(name='Spectral_Loss', data=spectral_loss, step=self.step_count)
            for i, (r_dist, e_dist) in enumerate(zip(real_samples, fake_samples)):
                tf.summary.histogram(name='Real_Distribution_{}'.format(i), data=r_dist, step=self.step_count)
                tf.summary.histogram(name='Encoder_Distribution_{}'.format(i), data=e_dist, step=self.step_count)

        return logits, all_q, all_log_q, all_p, all_log_p

    def sample(self, num_samples, temperature=1.0):

        # Get an initial random input
        z0_size = [num_samples, self.input_size//self.spatial_scaling, self.input_size//self.spatial_scaling, self.latent_channel_dim]
        dist = tfp.distributions.Normal(tf.zeros(z0_size), tf.ones(z0_size))
        z = dist.sample()

        index = 0
        s = self.prior_ftr0

        for cell in self.decoder.decoder_cells:
            if "combiner" in cell.name:
                if index > 0:
                    # Create latent space
                    sample = self.dec_sampler[index-1](s)
                    mu, log_sigma = tf.split(sample, 2, -1)
                    sigma = tf.math.exp(log_sigma)
                    dist = tfp.distributions.Normal(mu, sigma)
                    z = dist.sample()

                # Combine
                s = cell(s, z)
                index += 1

            else:
                s = cell(s)

        logits = self.image_conditional.image_conditional(s)
        # logits = self.image_conditional(s)

        return logits

    def discriminator_output(self, dists):
        discrimintor_outputs = []
        for dist, cell in zip(dists, self.discriminator.discriminator_cells):
            out = cell(dist)
            discrimintor_outputs.append(out)
        return discrimintor_outputs

    def decoder_output(self, logits):
        return tfp.distributions.Bernoulli(logits)

    def log_weight_norm(self, weight):
        # Returns log norm of the weight

        weight_norm = numpy.reshape(norm(weight, [1, 2, 3]), (-1, 1, 1, 1))
        log_weight_norm = tf.math.log(weight_norm)
        
        n = tf.math.exp(log_weight_norm)
        wn = numpy.reshape(norm(weight, [1, 2, 3]), (-1, 1, 1, 1))
        
        w = n*weight/(numpy.reshape(wn, (-1, 1, 1, 1)) + 1e-5)
        
        w = numpy.reshape(w, (-1, 1, 1, 1))

        return w

    def spectral_norm(self):
        # Taken from NVAE offical implementation, spectral loss helps with training                     

        weights = {}
        
        for l in self.all_conv_layers:
            weight = self.log_weight_norm(l.get_weights()[0])
            weight_mat = numpy.reshape(weight, (weight.shape[0], -1))
            if weight_mat.shape not in weights:
                weights[weight_mat.shape] = []
                
            weights[weight_mat.shape].append(weight_mat)
            
        loss = 0
        for i in weights:
            weights[i] = tf.stack(weights[i])
            num_iter = self.num_power_iter
            if i not in self.sr_u:
                num_w, row, col = weights[i].shape
                self.sr_u[i] = tf.math.l2_normalize(tf.random.normal([num_w, row]), axis=1)
                self.sr_v[i] = tf.math.l2_normalize(tf.random.normal([num_w, col]), axis=1)
                
                num_iter = 10*self.num_power_iter
                
            for j in range(num_iter):
                self.sr_v[i] = tf.math.l2_normalize(tf.squeeze(tf.linalg.matmul(tf.expand_dims(self.sr_u[i], 1), weights[i]), axis=1), axis=1)
                self.sr_u[i] = tf.math.l2_normalize(tf.squeeze(tf.linalg.matmul(weights[i], tf.expand_dims(self.sr_v[i], axis=2)), axis=2), axis=1)

            sigma = tf.linalg.matmul(tf.expand_dims(self.sr_u[i], axis=1), tf.linalg.matmul(weights[i], tf.expand_dims(self.sr_v[i], axis=2)))
            loss+= tf.reduce_sum(sigma)
        
        return loss

    def discriminator_loss(self, real_output, fake_output, weight = 1.0):
        disc_losses = []
        
        for real, fake in zip(real_output, fake_output):
            real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
            fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))
            disc_losses.append(weight*(tf.reduce_mean(real_loss)+tf.reduce_mean(fake_loss)))
        
        return disc_losses
    
    def autoencoder_loss(self, inputs, recon, weight = 1.0):
        # Workaround for an issue with tf.reduce_mean
        return weight * tf.reduce_sum(tf.square(inputs-recon))/tf.cast(tf.size(tf.square(inputs-recon)), tf.float32)
    
    def generator_loss(self, fake, weight = 1.0):
        return weight * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))
        

    def image2latent(self, x):
        """map image to latent space"""
        s = self.stem(2 * x - 1.0)
        for cell in self.encoder.encoder_cells:
            if (isinstance(cell, combinerEncoderCell)):
                pass
            else:
                s = cell(s)
        return s

    def encoder_zmeans(self, input_shape, latent_dim=2):
        """map image latent space (batch,8,8,128) to parameter space (batch,2) (ie, kappa as true value)"""
        input1 = tf.keras.layers.Input(shape=input_shape, name='image_latent_space')
        x = tf.keras.layers.Conv2D(64, (2, 2), strides=[2, 2], padding='valid', activation='relu')(input1)
        x = tf.keras.layers.Conv2D(128, (2, 2), strides=[2, 2], padding='valid', activation='relu')(x)
        x = tf.keras.layers.Reshape((-1, ))(x)
        output1 = tf.keras.layers.Dense(latent_dim, name='params_space')(x)

        encoder = tf.keras.models.Model(inputs=input1, outputs=output1, name='latent2params')
        opt = tf.keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='mse', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder

    def encoder_cls(self, input_shape, num_class=10):
        """map image latent space to parameter class(ic, kappa as class)"""
        input1 = tf.keras.layers.Input(shape=input_shape, name='image_latent_space')
        x = tf.keras.layers.Conv2D(64, (2, 2), strides=[2, 2], padding='valid', activation='relu')(input1)
        x = tf.keras.layers.Reshape((-1, ))(x)
        output1 = tf.keras.layers.Dense(num_class+1, activation='softmax', name='param_class')(x)

        encoder = tf.keras.models.Model(inputs=input1, outputs=output1, name='latent2class')
        opt = tf.keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
        encoder.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        encoder.summary()
        return encoder

