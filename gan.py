import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import os
import pathlib
import time
from matplotlib import pyplot as plt
from IPython import display

dataset = "ds_name"

link = "path to dataset"

path = dataset

list(path.parent.iterdir())

test = tf.io.read_file(str(path / 'n.jpg'))
test = tf.io.decode_jpeg(test)
plt.figure()
plt.imshow(test)

def initial(image_path):
    
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)

    
    
    
    w = tf.shape(image)[1]//2
    load_image = image[:, w:, :]
    ground_image = image[:, :w, :]

    
    load_image = tf.cast(load_image, tf.float32)
    ground_image = tf.cast(ground_image, tf.float32)

    return load_image, ground_image

inp, re = initial(str(path / 'n.jpg'))



buffer = 200

batch = 1

im_width = 256
im_height = 256

def res(load_image, ground_image, height, width):
    load_image = tf.image.res(load_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    ground_image = tf.image.res(ground_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return load_image, ground_image

def normalize(load_image, ground_image):
    load_image = (load_image / 127.5) - 1
    ground_image = (ground_image / 127.5) - 1

    return load_image, ground_image

def crop(load_image, ground_image):
    stacked_image = tf.stack([load_image, ground_image], axis=0)
    cropped_image = tf.image.crop(
        stacked_image, size=[2, im_height, im_width, 3])

    return cropped_image[0], cropped_image[1]




def jitter(load_image, ground_image):
    
    load_image, ground_image = res(load_image, ground_image, 286, 286)

    
    load_image, ground_image = crop(load_image, ground_image)

    if tf.random.uniform(()) > 0.5:
        
        load_image = tf.image.flip_left_right(load_image)
        ground_image = tf.image.flip_left_right(ground_image)

    return load_image, ground_image

def first_imagetrain(image_path):
    load_image, ground_image = initial(image_path)
    load_image, ground_image = jitter(load_image, ground_image)
    load_image, ground_image = normalize(load_image, ground_image)

    return load_image, ground_image

def first_imagetest(image_path):
    load_image, ground_image = initial(image_path)
    load_image, ground_image = res(load_image, ground_image,
                                    im_height, im_width)
    load_image, ground_image = normalize(load_image, ground_image)
    return load_image, ground_image

train_images = tf.data.Dataset.list_files(str(path / 'train/*.jpg'))
train_images = train_images.map(first_imagetrain,
                                num_parallel_calls=tf.data.AUTOTUNE)
train_images = train_images.shuffle(buffer)
train_images = train_images.batch(batch)


channels = 3

def down(filters, size):
    initializer = tf.random_normal_initializer(0.1, 0.05)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

down_model = down(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)

def up(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

up_model = up(3, 4)
up_result = up_model(down_result)

def Generator():
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
        down(64, 4, apply_batchnorm=False),  
        down(128, 4),  
        down(256, 4),  
        down(512, 4),  
        down(512, 4),  
        down(512, 4),  
        down(512, 4),  
        down(512, 4),  
    ]

    up_stack = [
        up(512, 4, apply_dropout=True),  
        up(512, 4, apply_dropout=True),  
        up(512, 4, apply_dropout=True),  
        up(512, 4),  
        up(256, 4),  
        up(128, 4),  
        up(64, 4),  
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(channels, 4,
                                            strides=2,
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  

    x = inputs

    
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])

LAMBDA = 100

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='load_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  

    down1 = down(64, 4, False)(x)  
    down2 = down(128, 4)(down1)  
    down3 = down(256, 4)(down2)  

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                    kernel_initializer=initializer,
                                    use_bias=False)(zero_pad1)  

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                    kernel_initializer=initializer)(zero_pad2)  

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['name[0]', 'name[1]', 'name[2]']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()

def train_step(load_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(load_image, training=True)

        disc_real_output = discriminator([load_image, target], training=True)
        disc_generated_output = discriminator([load_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

def fit(train_ds, test_ds, steps):
einput,target = next(iter(test_ds.take(1)))
train_step(load_image, target, step)


for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar)