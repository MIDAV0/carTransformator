from utilsNEW import *
import time
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device
import tensorflow as tf
from networksNEW import *
from copy import deepcopy
import neptune


class ALAE():
    def __init__(self, args):
        super(ALAE, self).__init__()

        self.model_name = 'ALAE'
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.augment_flag = args.augment_flag

        self.ds_iter = args.ds_iter
        self.iteration = args.iteration

        # self.gan_type = args.gan_type

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.lr = args.lr
        self.f_lr = args.f_lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        self.ema_decay = args.ema_decay

        """ Weight """
        self.adv_weight = args.adv_weight
        self.sty_weight = args.sty_weight
        self.ds_weight = args.ds_weight
        self.cyc_weight = args.cyc_weight

        self.r1_weight = args.r1_weight

        """ Generator """
        self.latent_dim = args.latent_dim
        self.style_dim = args.style_dim
        self.num_style = args.num_style

        """ Mapping Network """
        self.hidden_dim = args.hidden_dim

        self.checkpoint_dir = os.path.join(args.checkpoint_dir, self.model_dir)
        check_folder(self.checkpoint_dir)

        self.log_dir = os.path.join(args.log_dir, self.model_dir)
        check_folder(self.log_dir)


        # dataset_path = r'C:\Users\magil\Desktop\NMSC\CS\NEA\carsdataset'
        dataset_path = os.path.dirname(os.path.realpath('dataset'))

        self.dataset_path = os.path.join(dataset_path, 'train')
        self.test_dataset_path = os.path.join(dataset_path, 'test')
        self.domain_list = ['front', 'rear', 'side']
        self.num_domains = len(self.domain_list)


        print(self.dataset_path)
        print("domains", self.domain_list)
        print('num of domains', self.num_domains)

        print("##### Information #####")
        print("# dataset : ", self.dataset_name)
        print("# domain_list : ", self.domain_list)

        print("# batch_size : ", self.batch_size)
        print("# max iteration : ", self.iteration)
        print("# ds iteration : ", self.ds_iter)

        print()

        print("##### Generator #####")
        print("# latent_dim : ", self.latent_dim)
        print("# style_dim : ", self.style_dim)
        print("# num_style : ", self.num_style)

        print()

        print("##### Mapping Network #####")
        print("# hidden_dim : ", self.hidden_dim)

        print()

        print("##### Discriminator #####")

    ##################################################################################
    # Model
    ##################################################################################
    def build_model(self):

        """ Input Image"""
        img_class = Image_data(self.img_size, self.img_ch, self.domain_list, self.augment_flag)
        img_class.preprocess()

        dataset_num = len(img_class.images)
        print("Dataset number : ", dataset_num)

        img_and_domain = tf.data.Dataset.from_tensor_slices(
            (img_class.images, img_class.shuffle_images, img_class.domains))  # tensors with data. Will be 3 tensors

        gpu_device = '/gpu:0'

        img_and_domain = img_and_domain.shuffle(buffer_size=dataset_num,
                                                reshuffle_each_iteration=True).repeat()  # Randomly shuffles the elements of this dataset
        img_and_domain = img_and_domain.map(map_func=img_class.image_processing, num_parallel_calls=AUTOTUNE).batch(
            self.batch_size, drop_remainder=True)
        img_and_domain = img_and_domain.apply(
            prefetch_to_device(gpu_device, buffer_size=AUTOTUNE))  # Fits data based on device in use

        self.img_and_domain_iter = iter(img_and_domain)  # Iterator on data

        """ Network """
        self.generator = Generator(self.img_size, self.img_ch, self.style_dim, max_conv_dim=self.hidden_dim,
                                   name='Generator')
        self.mapping_network = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains,
                                              name='MappingNetwork')
        self.style_encoder = StyleEncoder(self.img_size, self.style_dim, self.num_domains,
                                          max_conv_dim=self.hidden_dim, name='StyleEncoder')
        self.discriminator = Discriminator(self.img_size, self.num_domains, max_conv_dim=self.hidden_dim,
                                           name='Discriminator')

        """ Network for testing"""
        self.generator_ema = deepcopy(self.generator)
        self.mapping_network_ema = deepcopy(self.mapping_network)
        self.style_encoder_ema = deepcopy(self.style_encoder)

        """ Finalize model """
        x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)  # image
        y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)  # labels  (true or fake)
        z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)  # latent space
        s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)  # style

        _ = self.mapping_network([z, y])
        _ = self.mapping_network_ema([z, y])
        _ = self.style_encoder([x, y])
        _ = self.style_encoder_ema([x, y])
        _ = self.generator([x, s])
        _ = self.generator_ema([x, s])
        _ = self.discriminator([x, y])

        """ Optimizer """
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2,
                                                    epsilon=1e-08)
        self.e_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2,
                                                    epsilon=1e-08)
        self.f_optimizer = tf.keras.optimizers.Adam(learning_rate=self.f_lr, beta_1=self.beta1, beta_2=self.beta2,
                                                    epsilon=1e-08)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, beta_1=self.beta1, beta_2=self.beta2,
                                                    epsilon=1e-08)

        """ Checkpoint """
        self.ckpt = tf.train.Checkpoint(generator=self.generator, generator_ema=self.generator_ema,
                                        mapping_network=self.mapping_network,
                                        mapping_network_ema=self.mapping_network_ema,
                                        style_encoder=self.style_encoder, style_encoder_ema=self.style_encoder_ema,
                                        discriminator=self.discriminator,
                                        g_optimizer=self.g_optimizer, e_optimizer=self.e_optimizer,
                                        f_optimizer=self.f_optimizer,
                                        d_optimizer=self.d_optimizer)
        # groups trainable objects, saving and restoring them

        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir,
                                                  max_to_keep=1)  # saves checkpoints to the directory
        self.start_iteration = 0

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            self.start_iteration = int(self.manager.latest_checkpoint.split('-')[-1])
            print('Latest checkpoint restored')
            print('start iteration : ', self.start_iteration)
        else:
            print('Not restoring from saved checkpoint')


    # x_real - real image, z_trg = latent space, y_trg - domain, x_ref - reference img, s_trg - style,
    @tf.function
    def g_train_step(self, x_real, y_org, y_trg, z_trgs=None, x_refs=None):
        with tf.GradientTape(persistent=True) as g_tape:
            if z_trgs is not None:
                z_trg, z_trg2 = z_trgs
            if x_refs is not None:
                x_ref, x_ref2 = x_refs

            # adversarial loss  Min Max loss
            if z_trgs is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else:
                s_trg = self.style_encoder([x_ref, y_trg])

            x_fake = self.generator([x_real, s_trg])
            fake_logit = self.discriminator([x_fake, y_trg])
            g_adv_loss = self.adv_weight * generator_loss(fake_logit)

            # style reconstruction loss
            s_pred = self.style_encoder([x_fake, y_trg])
            g_sty_loss = self.sty_weight * L1_loss(s_pred, s_trg)

            # diversity sensitive loss
            if z_trgs is not None:
                s_trg2 = self.mapping_network([z_trg2, y_trg])
            else:
                s_trg2 = self.style_encoder([x_ref2, y_trg])

            x_fake2 = self.generator([x_real, s_trg2])
            x_fake2 = tf.stop_gradient(x_fake2)
            g_ds_loss = -self.ds_weight * L1_loss(x_fake, x_fake2)

            # cycle-consistency loss
            s_org = self.style_encoder([x_real, y_org])
            x_rec = self.generator([x_fake, s_org])
            g_cyc_loss = self.cyc_weight * L1_loss(x_rec, x_real)

            regular_loss = regularization_loss(self.generator)

            g_loss = g_adv_loss + g_sty_loss + g_ds_loss + g_cyc_loss + regular_loss

        g_train_variable = self.generator.trainable_variables
        g_gradient = g_tape.gradient(g_loss, g_train_variable)
        self.g_optimizer.apply_gradients(zip(g_gradient, g_train_variable))

        if z_trgs is not None:
            f_train_variable = self.mapping_network.trainable_variables
            e_train_variable = self.style_encoder.trainable_variables

            f_gradient = g_tape.gradient(g_loss, f_train_variable)
            e_gradient = g_tape.gradient(g_loss, e_train_variable)

            self.f_optimizer.apply_gradients(zip(f_gradient, f_train_variable))
            self.e_optimizer.apply_gradients(zip(e_gradient, e_train_variable))

        return g_adv_loss, g_sty_loss, g_ds_loss, g_cyc_loss, g_loss

    @tf.function
    def d_train_step(self, x_real, y_org, y_trg, z_trg=None, x_ref=None):
        with tf.GradientTape() as d_tape:

            if z_trg is not None:
                s_trg = self.mapping_network([z_trg, y_trg])
            else:  # x_ref is not None
                s_trg = self.style_encoder([x_ref, y_trg])

            x_fake = self.generator([x_real, s_trg])

            real_logit = self.discriminator([x_real, y_org])
            fake_logit = self.discriminator([x_fake, y_trg])

            d_adv_loss = self.adv_weight * discriminator_loss(real_logit, fake_logit)

            d_adv_loss += self.r1_weight * r1_gp_req(self.discriminator, x_real, y_org)

            regular_loss = regularization_loss(self.discriminator)

            d_loss = d_adv_loss + regular_loss

        d_train_variable = self.discriminator.trainable_variables
        d_gradient = d_tape.gradient(d_loss, d_train_variable)
        self.d_optimizer.apply_gradients(zip(d_gradient, d_train_variable))

        return d_adv_loss, d_loss

    def train(self):

        start_time = time.time()

        # setup neptune
        neptune.init('midav/sandbox',
                     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiYzI4NjUxZGEtMmRmYi00NGNiLWI4YWQtYmZiMzk3YWQ3YzExIn0=")
        neptune.create_experiment(name='ALAE')

        print("Neptune is connected")

        # setup tensorboards
        train_summary_writer = tf.summary.create_file_writer(self.log_dir)
        ds_weight_init = self.ds_weight

        for idx in range(self.start_iteration, self.iteration):
            iter_start_time = time.time()

            # decay weight for diversity sensitive loss
            if self.ds_weight > 0:
                self.ds_weight = ds_weight_init - (ds_weight_init / self.ds_iter) * idx  # 0.999, 0.998, 0.997

            x_real, _, y_org = next(self.img_and_domain_iter)  # outputs next item of the iterator
            x_ref, x_ref2, y_trg = next(self.img_and_domain_iter)

            z_trg = tf.random.normal(shape=[self.batch_size, self.latent_dim])
            z_trg2 = tf.random.normal(shape=[self.batch_size, self.latent_dim])

            # update discriminator
            # latent space
            d_adv_loss_latent, d_loss_latent = self.d_train_step(x_real, y_org, y_trg, z_trg=z_trg)
            # style encoder
            d_adv_loss_ref, d_loss_ref = self.d_train_step(x_real, y_org, y_trg, x_ref=x_ref)

            # update generator
            # latent space
            g_adv_loss_latent, g_sty_loss_latent, g_ds_loss_latent, g_cyc_loss_latent, g_loss_latent = self.g_train_step(
                x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2])
            # style encoder
            g_adv_loss_ref, g_sty_loss_ref, g_ds_loss_ref, g_cyc_loss_ref, g_loss_ref = self.g_train_step(x_real, y_org,
                                                                                                          y_trg,
                                                                                                          x_refs=[x_ref,
                                                                                                                  x_ref2])

            # compute moving average of network parameters
            # updates weights of test network using train network
            moving_average(self.generator, self.generator_ema, beta=self.ema_decay)
            moving_average(self.mapping_network, self.mapping_network_ema, beta=self.ema_decay)
            moving_average(self.style_encoder, self.style_encoder_ema, beta=self.ema_decay)

            if idx == 0:
                g_params = self.generator.count_params()  # count number of trainable parameters
                d_params = self.discriminator.count_params()
                print("G network parameters : ", format(g_params, ','))
                print("D network parameters : ", format(d_params, ','))
                print("Total network parameters : ", format(g_params + d_params, ','))

            # save to neptune

            neptune.log_metric('g/latent/adv_loss', g_adv_loss_latent)
            neptune.log_metric('g/latent/sty_loss', g_sty_loss_latent)
            neptune.log_metric('g/latent/ds_loss', g_ds_loss_latent)
            neptune.log_metric('g/latent/cyc_loss', g_cyc_loss_latent)
            neptune.log_metric('g/latent/loss', g_loss_latent)

            neptune.log_metric('g/ref/adv_loss', g_adv_loss_ref)
            neptune.log_metric('g/ref/sty_loss', g_sty_loss_ref)
            neptune.log_metric('g/ref/ds_loss', g_ds_loss_ref)
            neptune.log_metric('g/ref/cyc_loss', g_cyc_loss_ref)
            neptune.log_metric('g/ref/loss', g_loss_ref)

            neptune.log_metric('g/ds_weight', self.ds_weight)

            neptune.log_metric('d/latent/adv_loss', d_adv_loss_latent)
            neptune.log_metric('d/latent/loss', d_loss_latent)

            neptune.log_metric('d/latent/loss', d_adv_loss_ref)
            neptune.log_metric('d/ref/loss', d_loss_ref)

            # save to tensotboard
            with train_summary_writer.as_default():
                tf.summary.scalar('g/latent/adv_loss', g_adv_loss_latent, step=idx)
                tf.summary.scalar('g/latent/sty_loss', g_sty_loss_latent, step=idx)
                tf.summary.scalar('g/latent/ds_loss', g_ds_loss_latent, step=idx)
                tf.summary.scalar('g/latent/cyc_loss', g_cyc_loss_latent, step=idx)
                tf.summary.scalar('g/latent/loss', g_loss_latent, step=idx)

                tf.summary.scalar('g/ref/adv_loss', g_adv_loss_ref, step=idx)
                tf.summary.scalar('g/ref/sty_loss', g_sty_loss_ref, step=idx)
                tf.summary.scalar('g/ref/ds_loss', g_ds_loss_ref, step=idx)
                tf.summary.scalar('g/ref/cyc_loss', g_cyc_loss_ref, step=idx)
                tf.summary.scalar('g/ref/loss', g_loss_ref, step=idx)

                tf.summary.scalar('g/ds_weight', self.ds_weight, step=idx)

                tf.summary.scalar('d/latent/adv_loss', d_adv_loss_latent, step=idx)
                tf.summary.scalar('d/latent/loss', d_loss_latent, step=idx)

                tf.summary.scalar('d/ref/adv_loss', d_adv_loss_ref, step=idx)
                tf.summary.scalar('d/ref/loss', d_loss_ref, step=idx)

            # save every self.save_freq
            if np.mod(idx + 1, self.save_freq) == 0:  # np.mod computes a reminder
                self.manager.save(checkpoint_number=idx + 1)

            print("iter: [%6d/%6d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                idx, self.iteration, time.time() - iter_start_time, d_loss_latent + d_loss_ref,
                g_loss_latent + g_loss_ref))

        # save model for final step
        self.manager.save(checkpoint_number=self.iteration)

        print("Total train time: %4.4f" % (time.time() - start_time))

    @property
    def model_dir(self):
        return "{}_{}".format(self.model_name, self.dataset_name)

