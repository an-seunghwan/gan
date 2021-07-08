#%%
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
print('TensorFlow version:', tf.__version__)
print('Eager Execution Mode:', tf.executing_eagerly())
print('available GPU:', tf.config.list_physical_devices('GPU'))
from tensorflow.python.client import device_lib
print('==========================================')
print(device_lib.list_local_devices())
# tf.debugging.set_log_device_placement(False)
#%%
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
# os.chdir('D:\gan')
os.chdir('/Users/anseunghwan/Documents/GitHub/gan')

from modules import model_cmnist3
#%%
PARAMS = {
    "batch_size": 64,
    "epochs": 10000, 
    "learning_rate": 0.0001, 
    "data": "cmnist",
    "class_num": 10,
    "latent_dim": 128, 
    "embedding_dim": 32, 
    "embedding_dim_mult": (2, 4),
    # "ema_rate": 0.999
}

print(PARAMS)
#%%
if PARAMS['data'] == "cmnist":
    (x_train, y_train), (x_test, y_test) = K.datasets.mnist.load_data()
    
    '''colored mnist'''
    PARAMS['data_dim'] = 32
    PARAMS['n_layer'] = int(np.log(PARAMS['data_dim']) / np.log(2)) - 1
    PARAMS["channel"] = 3

    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    from tensorflow.keras.utils import to_categorical
    y_train_onehot = to_categorical(y_train, num_classes=PARAMS['class_num'])
    y_test_onehot = to_categorical(y_test, num_classes=PARAMS['class_num'])

    np.random.seed(1)
    def colored_mnist(image):
        image = tf.image.resize(image, [PARAMS['data_dim'], PARAMS['data_dim']], method='nearest')

        # edge detection
        image = cv2.Canny(image.numpy(), 10., 255.)
        image[np.where(image > 0)] = 1.
        image[np.where(image <= 0)] = 0.
        # plt.imshow(laplacian)

        # width
        dilation_size = np.random.choice(np.arange(6), 1)[0]
        kernel = np.ones((dilation_size, dilation_size))
        image = cv2.dilate(image, kernel)
        # plt.imshow(laplacian)

        # color
        color = np.random.uniform(0., 1., 3)
        color = color / np.linalg.norm(color)
        image = image[..., tf.newaxis] * color[tf.newaxis, tf.newaxis, :]
        # plt.imshow(laplacian)

        # size
        size = np.random.uniform(0.4, 0.5, 1)
        downsize = int(PARAMS['data_dim'] * size) * 2
        image = tf.image.resize(image, [downsize, downsize], method='nearest')
        padding = int((PARAMS['data_dim'] - downsize) / 2)
        image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), 'constant') 
        # plt.imshow(laplacian)
        
        assert image.shape == (PARAMS['data_dim'], PARAMS['data_dim'], PARAMS['channel'])
        
        return tf.cast(image, tf.float32)

    cx_train = []
    for i in tqdm(range(len(x_train)), desc='generating colored mnist'):
        cx_train.append(colored_mnist(x_train[i]))
    cx_train = np.array(cx_train)
    # plt.imshow(cx_train[0])
    
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i in range(10):
        axes.flatten()[i].imshow(cx_train[i])
        axes.flatten()[i].axis('off')
    plt.savefig('./assets/data_examples.png',
                dpi=200, bbox_inches="tight", pad_inches=0.1)
    # plt.show()
    plt.close()

    train_dataset = tf.data.Dataset.from_tensor_slices((cx_train, y_train_onehot)).shuffle(len(cx_train), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
else:
    print('Invalid data type!')
    assert 0 == 1
#%%
encoder = model_cmnist3.build_encoder(PARAMS)
encoder.summary()
generator = model_cmnist3.build_generator(PARAMS)
generator.summary()
img_discriminator = model_cmnist3.build_image_discriminator(PARAMS)
img_discriminator.summary()
z_discriminator = model_cmnist3.build_z_discriminator(PARAMS)
z_discriminator.summary()
classifier = model_cmnist3.build_image_classifier(PARAMS)
classifier.summary()
#%%
@tf.function
def loss_function(x_batch, y_batch, PARAMS):
    '''generation'''
    z = tf.random.normal([PARAMS['batch_size'], PARAMS['latent_dim']])

    epsilon = tf.random.normal([PARAMS['batch_size'], PARAMS['latent_dim']])
    recon_z = encoder([x_batch, epsilon])

    noise = np.random.uniform(0.0, 1.0, [PARAMS['batch_size'], PARAMS['data_dim'], PARAMS['data_dim'], 1]).astype('float32')
    generated_images = generator([z]*PARAMS['n_layer'] + [noise] + [y_batch])
    reconstructed_images = generator([recon_z]*PARAMS['n_layer'] + [noise] + [y_batch])
    
    '''loss'''    
    lambda_coef = 0.2

    # 1. encoder
    encoder_loss1 = lambda_coef * tf.reduce_mean(tf.reduce_sum(tf.abs(x_batch - reconstructed_images), axis=[1,2,3]))
    encoder_loss2 = tf.reduce_mean(-tf.math.log(z_discriminator(recon_z) + 1e-8) + tf.math.log(1 - z_discriminator(recon_z) + 1e-8))
    encoder_loss = encoder_loss1 + encoder_loss2

    # 2. generator
    generator_loss1 = lambda_coef * tf.reduce_mean(tf.reduce_sum(tf.abs(x_batch - reconstructed_images), axis=[1,2,3]))
    generator_loss2 = tf.reduce_mean(-tf.math.log(img_discriminator(reconstructed_images)+1e-8) + tf.math.log(1 - img_discriminator(reconstructed_images) + 1e-8))
    generator_loss = generator_loss1 + generator_loss2

    # 3. discriminator of image
    img_dis_loss1 = 2 * tf.reduce_mean(-tf.math.log(img_discriminator(x_batch) + 1e-8))
    img_dis_loss2 = tf.reduce_mean(-tf.math.log(1 - img_discriminator(reconstructed_images) + 1e-8))
    img_dis_loss3 = tf.reduce_mean(-tf.math.log(1 - img_discriminator(generated_images) + 1e-8))
    img_dis_loss = img_dis_loss1 + img_dis_loss2 + img_dis_loss3

    # 4. discriminator of z
    z_dis_loss1 = tf.reduce_mean(-tf.math.log(z_discriminator(z) + 1e-8))
    z_dis_loss2 = tf.reduce_mean(-tf.math.log(1 - z_discriminator(recon_z) + 1e-8))
    z_dis_loss = z_dis_loss1 + z_dis_loss2 

    # 5. classifier
    classification_loss1 = tf.reduce_mean(-tf.math.log(tf.reduce_sum(classifier(x_batch) * y_batch, axis=-1) + 1e-8))
    classification_loss2 = tf.reduce_mean(-tf.math.log(tf.reduce_sum(classifier(reconstructed_images) * y_batch, axis=-1) + 1e-8))
    classification_loss = classification_loss1 + classification_loss2
    
    return [z, recon_z, generated_images, reconstructed_images], [encoder_loss, generator_loss, img_dis_loss, z_dis_loss, classification_loss]
#%%
encoder_optimizer = K.optimizers.Adam(PARAMS['learning_rate'])
generator_optimizer = K.optimizers.Adam(PARAMS['learning_rate'])
img_discriminator_optimizer = K.optimizers.Adam(PARAMS['learning_rate'])
z_discriminator_optimizer = K.optimizers.Adam(PARAMS['learning_rate'])
classifier_optimizer = K.optimizers.Adam(PARAMS['learning_rate'])

# ema = tf.train.ExponentialMovingAverage(decay=PARAMS['ema_rate'])
#%%
# @tf.function
# def train_one_step(x_batch, y_batch, PARAMS):
#     with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as img_dis_tape, tf.GradientTape() as z_dis_tape, tf.GradientTape() as cls_tape:

#         [z, recon_z, generated_images, reconstructed_images], [encoder_loss, generator_loss, img_dis_loss, z_dis_loss, classification_loss] = loss_function(x_batch, y_batch, PARAMS)
        
#     gradients_of_encoder = enc_tape.gradient(encoder_loss, encoder.trainable_variables)
#     gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
#     gradients_of_img_discriminator = img_dis_tape.gradient(img_dis_loss, img_discriminator.trainable_variables)
#     gradients_of_z_discriminator = z_dis_tape.gradient(z_dis_loss, z_discriminator.trainable_variables)
#     gradients_of_classifier = cls_tape.gradient(classification_loss, classifier.trainable_variables)

#     encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
#     generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     img_discriminator_optimizer.apply_gradients(zip(gradients_of_img_discriminator, img_discriminator.trainable_variables))
#     z_discriminator_optimizer.apply_gradients(zip(gradients_of_z_discriminator, z_discriminator.trainable_variables))
#     classifier_optimizer.apply_gradients(zip(gradients_of_classifier, classifier.trainable_variables))
    
#     # ema.apply(encoder.trainable_variables)
#     # ema.apply(generator.trainable_variables)
#     # ema.apply(img_discriminator.trainable_variables)
#     # ema.apply(z_discriminator.trainable_variables)
#     # ema.apply(classifier.trainable_variables)
    
#     return [z, recon_z, generated_images, reconstructed_images], [encoder_loss, generator_loss, img_dis_loss, z_dis_loss, classification_loss]
#%%
step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
def generate_and_save_images(images, epochs):
    fig = plt.figure(figsize=(4,4))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('./assets/image_at_epoch_{:04d}.png'.format(epochs))
    # plt.show()
    plt.close()
#%%
'''training'''
for _ in progress_bar:
    x_batch, y_batch = next(iter(train_dataset))
    step += 1
    
    # [z, recon_z, generated_images, reconstructed_images], [encoder_loss, generator_loss, img_dis_loss, z_dis_loss, classification_loss] = train_one_step(x_batch, y_batch, PARAMS)
    with tf.GradientTape() as enc_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as img_dis_tape, tf.GradientTape() as z_dis_tape, tf.GradientTape() as cls_tape:

        [z, recon_z, generated_images, reconstructed_images], [encoder_loss, generator_loss, img_dis_loss, z_dis_loss, classification_loss] = loss_function(x_batch, y_batch, PARAMS)
        
        eps = tf.random.uniform(shape=[PARAMS['batch_size'], 1, 1, 1])
        x_hat = eps*x_batch + (1 - eps)*generated_images
        
        with tf.GradientTape() as t:
            t.watch(x_hat)
            d_hat = img_discriminator(x_hat)

        gradients = t.gradient(d_hat, [x_hat]) 
        l2_norm = tf.math.sqrt(tf.reduce_sum(tf.math.square(gradients[0]), axis=[1,2,3]))
        gradient_penalty = tf.reduce_mean(tf.math.square(l2_norm - 1.))
        img_dis_loss += 0.5 * gradient_penalty
        
    gradients_of_encoder = enc_tape.gradient(encoder_loss, encoder.trainable_variables)
    gradients_of_generator = gen_tape.gradient(generator_loss, generator.trainable_variables)
    gradients_of_img_discriminator = img_dis_tape.gradient(img_dis_loss, img_discriminator.trainable_variables)
    gradients_of_z_discriminator = z_dis_tape.gradient(z_dis_loss, z_discriminator.trainable_variables)
    gradients_of_classifier = cls_tape.gradient(classification_loss, classifier.trainable_variables)

    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    img_discriminator_optimizer.apply_gradients(zip(gradients_of_img_discriminator, img_discriminator.trainable_variables))
    z_discriminator_optimizer.apply_gradients(zip(gradients_of_z_discriminator, z_discriminator.trainable_variables))
    classifier_optimizer.apply_gradients(zip(gradients_of_classifier, classifier.trainable_variables))
    
    progress_bar.set_description('setting: {} | iteration {}/{} | enc loss {:.3f}, gen loss {:.3f}, img loss {:.3f}, z loss {:.3f}, cls loss {:.3f}'.format(
        PARAMS['data'], 
        step, PARAMS['epochs'], 
        encoder_loss.numpy(), generator_loss.numpy(), img_dis_loss.numpy(), z_dis_loss.numpy(), classification_loss.numpy()
    ))

    if step % 50 == 0:
        generate_and_save_images(reconstructed_images, step)

    if step == PARAMS['epochs']: break
#%%
encoder.save_weights('./assets/weights_enc/weights')
generator.save_weights('./assets/weights_gen/weights')
img_discriminator.save_weights('./assets/weights_img/weights')
z_discriminator.save_weights('./assets/weights_z/weights')
classifier.save_weights('./assets/weights_cls/weights')
#%%