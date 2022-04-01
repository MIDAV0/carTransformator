from utils import *
from networks import *
import tensorflow as tf

import numpy as np
from PIL import Image as pilImage
import cv2



""" Test """
""" Network """
class only():
    def __init__(self):
        self.img_size = 384
        self.img_ch = 3
        self.style_dim = 64
        self.hidden_dim = 512
        self.num_domains = 3
        self.batch_size = 4
        self.latent_dim = 16
        self.checkpoint_dir = r'C:\Users\magil\Desktop\NMSC\CS\NEA\GAN model setup\checkpoint\StarGAN_v2_carsdataset'

        self.generator_ema = Generator(self.img_size, self.img_ch, self.style_dim, max_conv_dim=self.hidden_dim,
                                       name='Generator')
        self.mapping_network_ema = MappingNetwork(self.style_dim, self.hidden_dim, self.num_domains, name='MappingNetwork')
        self.style_encoder_ema = StyleEncoder(self.img_size, self.style_dim, self.num_domains, max_conv_dim=self.hidden_dim,
                                              name='StyleEncoder')

        """ Finalize model (build) """
        x = np.ones(shape=[self.batch_size, self.img_size, self.img_size, self.img_ch], dtype=np.float32)
        y = np.ones(shape=[self.batch_size, 1], dtype=np.int32)
        z = np.ones(shape=[self.batch_size, self.latent_dim], dtype=np.float32)
        s = np.ones(shape=[self.batch_size, self.style_dim], dtype=np.float32)

        _ = self.mapping_network_ema([z, y])
        _ = self.style_encoder_ema([x, y])
        _ = self.generator_ema([x, s])

        """ Checkpoint """
        self.ckpt = tf.train.Checkpoint(generator_ema=self.generator_ema,
                                        mapping_network_ema=self.mapping_network_ema,
                                        style_encoder_ema=self.style_encoder_ema)
        self.manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_dir, max_to_keep=1)

        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print('Latest checkpoint restored!!')
        else:
            print('Not restoring from saved checkpoint')

    def load_image_custom(self, image_path, img_size, img_channel):
        x = tf.io.read_file(image_path)
        x_decode = tf.image.decode_jpeg(x, channels=img_channel, dct_method='INTEGER_ACCURATE')
        img = tf.image.resize(x_decode, [img_size, img_size])
        img = preprocess_fit_train_image(img)

        return img

    def random_latent(self, src_img_path, domain):


        src_img = self.load_image_custom(src_img_path, self.img_size, self.img_ch)  # [img_size, img_size, img_ch]
        src_img = tf.expand_dims(src_img, axis=0)

        src_img = tf.expand_dims(src_img[0], axis=0)
        #domain.sort()
        domain_fix = tf.constant(domain)

        z_trgs = tf.random.normal(shape=[10, self.latent_dim])


        n = 0
        for dom in list(domain_fix):
            for row in range(10):
                z_trg = tf.expand_dims(z_trgs[row], axis=0)

                y_trg = tf.reshape(dom, shape=[1, 1])
                s_trg = self.mapping_network_ema([z_trg, y_trg])
                x_fake = self.generator_ema([src_img, s_trg])
                x_fake = postprocess_images(x_fake)

                col_image = x_fake[0]
                ima = pilImage.fromarray(np.uint8(col_image), 'RGB')
                n +=1
                ima.save(str(n)+'ima.jpg')


imaaaa = only()
img_src = r'C:\Users\magil\Desktop\NMSC\CS\NEA\carsdataset\test\ref_imgs\front\13609Car.jpg'

imaaaa.random_latent(img_src, [0, 1, 2])