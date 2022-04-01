import tensorflow as tf
import sqlite3
import random
from tensorflow.python.data.experimental import AUTOTUNE, prefetch_to_device

class Image_data:

    def __init__(self, img_size, channels, domain_list, augment_flag):
        self.img_height = img_size
        self.img_width = img_size
        self.channels = channels
        self.augment_flag = augment_flag

        self.domain_list = domain_list

        self.images = []
        self.shuffle_images = []
        self.domains = []

    def image_processing(self, filename, filename2, domain):

        # x = tf.io.read_file(filename)
        x = tf.convert_to_tensor(filename)
        x_resh = tf.reshape(x, [])
        x_decode = tf.image.decode_jpeg(x_resh, channels=self.channels,
                                        dct_method='INTEGER_ACCURATE')  # encodes img to uint8 type
        img = tf.image.resize(x_decode, [self.img_height, self.img_width])
        img = preprocess_fit_train_image(img)

        # x = tf.io.read_file(filename2)
        x = tf.convert_to_tensor(filename2)
        x_resh = tf.reshape(x, [])
        x_decode = tf.image.decode_jpeg(x_resh, channels=self.channels, dct_method='INTEGER_ACCURATE')
        img2 = tf.image.resize(x_decode, [self.img_height, self.img_width])
        img2 = preprocess_fit_train_image(img2)

        if self.augment_flag:
            seed = random.randint(0, 2 ** 31 - 1)
            condition = tf.greater_equal(tf.random.uniform(shape=[], minval=0.0, maxval=1.0), 0.5)

            augment_height_size = self.img_height + int(self.img_height * 0.1)
            augment_width_size = self.img_width + int(self.img_width * 0.1)

            img = tf.cond(pred=condition,
                          true_fn=lambda: augmentation(img, augment_height_size, augment_width_size, seed),
                          false_fn=lambda: img)

            img2 = tf.cond(pred=condition,
                           true_fn=lambda: augmentation(img2, augment_height_size, augment_width_size, seed),
                           false_fn=lambda: img2)

        return img, img2, domain

    def preprocess(self):
        # self.domain_list = ['front', 'rear', 'side']
        conn = sqlite3.connect(r'C:\Users\magil\Desktop\NMSC\CS\NEA\GUI\carsTrainDatabase.db')

        for idx, domain in enumerate(self.domain_list):
            image_list = []
            shuffle_list = []
            domain_list = []
            if domain == 'front':
                cursor = conn.execute("SELECT * FROM carss WHERE position='front'")
                rows = cursor.fetchall()
                for ind, row in enumerate(rows):
                    image_list.append(row[0])

                shuffle_list = random.sample(image_list, len(image_list))
                domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]
                self.images.extend(image_list)
                self.shuffle_images.extend(shuffle_list)
                self.domains.extend(domain_list)

            if domain == 'rear':
                cursor = conn.execute("SELECT * FROM carss WHERE position='rear'")
                rows = cursor.fetchall()
                for ind, row in enumerate(rows):
                    image_list.append(row[0])
                shuffle_list = random.sample(image_list, len(image_list))
                domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]
                self.images.extend(image_list)
                self.shuffle_images.extend(shuffle_list)
                self.domains.extend(domain_list)

            if domain == 'side':
                cursor = conn.execute("SELECT * FROM carss WHERE position='side'")
                rows = cursor.fetchall()
                for ind, row in enumerate(rows):
                    image_list.append(row[0])

                shuffle_list = random.sample(image_list, len(image_list))
                domain_list = [[idx]] * len(image_list)  # [ [0], [0], ... , [0] ]
                self.images.extend(image_list)
                self.shuffle_images.extend(shuffle_list)
                self.domains.extend(domain_list)
                image_list = []
                shuffle_list = []
                domain_list = []
        conn.close()


def adjust_dynamic_range(images, range_in, range_out, out_dtype):
    scale = (range_out[1] - range_out[0]) / (range_in[1] - range_in[0])
    bias = range_out[0] - range_in[0] * scale
    images = images * scale + bias
    images = tf.clip_by_value(images, range_out[0], range_out[1]) # clips value between 0 and 255
    images = tf.cast(images, dtype=out_dtype)  # casts a tensor to a new type
    return images

def preprocess_fit_train_image(images):
    images = adjust_dynamic_range(images, range_in=(0.0, 255.0), range_out=(-1.0, 1.0), out_dtype=tf.dtypes.float32)
    return images









domain_list=['front', 'rear', 'side']

img_class = Image_data(384, 3, domain_list, augment_flag=True)
img_class.preprocess()

dataset_num = len(img_class.images)
print("Dataset number : ", dataset_num)

img_and_domain = tf.data.Dataset.from_tensor_slices((img_class.images, img_class.shuffle_images, img_class.domains))  # tensors with data. Will be 3 tensors

img_and_domain = img_and_domain.shuffle(buffer_size=dataset_num, reshuffle_each_iteration=True).repeat()  # Randomly shuffles the elements of this dataset. part of tensorflow library
img_and_domain = img_and_domain.map(map_func=img_class.image_processing, num_parallel_calls=AUTOTUNE).batch(self.batch_size, drop_remainder=True)
