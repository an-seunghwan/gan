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
import time
os.chdir('/home/jeon/Desktop/an/gan')

from modules import model
#%%
PARAMS = {
    "batch_size": 10,
    "epochs": 10000, 
    "learning_rate": 0.0001, 
    "data": "celeba64",
    "data_dim": 64, 
    "channel": 3,
    "latent_dim": 512, 
    "n_layer": 5,
    "embedding_dim": 32, 
    "embedding_dim_mult": (1, 2, 2, 2, 4),
    "num_res_blocks": 3, 
    "attn_resolutions": (32, ),
    "ema_rate": 0.999
}

print(PARAMS)
#%%
import tensorflow_datasets as tfds
# (train_images, test_images), info = tfds.load('celeb_a', split=['train', 'test'], shuffle_files=False, with_info=True)
(train_images, test_images), info = tfds.load('celeb_a', split=['train[:10%]', 'test[:10%]'], shuffle_files=False, with_info=True)

print('image shape:', info.features['image'].shape)
PARAMS['data_dim'] = 64
attr_dict = {x:i for i, x in enumerate(info.features["attributes"].keys())}
print('attribute:', len(attr_dict))
PARAMS['attr_num'] = len(attr_dict)

eye = np.eye(len(attr_dict))
@tf.function
def normalize(data):    
    attr = data['attributes']
    idx = [attr_dict.get(x) for x in attr.keys() if attr.get(x) is True]
    tag = np.sum(eye[idx], axis=0) # one-hot vector
    image = (tf.cast(data['image'], tf.float32) - 127.5) / 127.5
    image = tf.image.resize(image, [PARAMS['data_dim'], PARAMS['data_dim']], method='nearest')
    return image, tag

train_dataset = train_images.map(normalize).shuffle(len(train_images), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
test_dataset = test_images.map(normalize).shuffle(len(test_images), reshuffle_each_iteration=True).batch(PARAMS['batch_size'])
#%%
encoder = model.build_encoder(PARAMS)
encoder.summary()
generator = model.build_generator(PARAMS)
generator.summary()
img_discriminator = model.build_image_discriminator(PARAMS)
img_discriminator.summary()
z_discriminator = model.build_z_discriminator(PARAMS)
z_discriminator.summary()
#%%
cross_entropy = K.losses.BinaryCrossentropy(from_logits=True)
def img_discriminator_loss(real_decision, fake_decision1, fake_decision2):
    real_loss = cross_entropy(tf.ones_like(real_decision), real_decision)
    fake_loss1 = cross_entropy(tf.zeros_like(fake_decision1), fake_decision1)
    fake_loss2 = cross_entropy(tf.zeros_like(fake_decision2), fake_decision2)
    total_loss = real_loss + fake_loss1 + fake_loss2
    return total_loss

def z_discriminator_loss(real_decision, fake_decision):
    real_loss = cross_entropy(tf.ones_like(real_decision), real_decision)
    fake_loss = cross_entropy(tf.zeros_like(fake_decision), fake_decision)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_decision):
    return cross_entropy(tf.ones_like(fake_decision), fake_decision)

def encoder_loss(fake_decision):
    return cross_entropy(tf.ones_like(fake_decision), fake_decision)
#%%
encoder_optimizer = K.optimizers.Adam(0.001)
generator_optimizer = K.optimizers.Adam(0.001)
img_discriminator_optimizer = K.optimizers.Adam(0.001)
z_discriminator_optimizer = K.optimizers.Adam(0.001)

ema = tf.train.ExponentialMovingAverage(decay=PARAMS['ema_rate'])
#%%
@tf.function
def train_one_step(x_batch, y_batch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as img_dis_tape, tf.GradientTape() as z_dis_tape, tf.GradientTape() as enc_tape:
        z = tf.random.normal([PARAMS['batch_size'], PARAMS['latent_dim']])
        recon_z = encoder(x_batch)
        noise = np.random.uniform(0.0, 1.0, [PARAMS['batch_size'], PARAMS['data_dim'], PARAMS['data_dim'], 1]).astype('float32')
        generated_images = generator([z]*PARAMS['n_layer'] + [noise] + [y_batch])
        reconstructed_images = generator([recon_z]*PARAMS['n_layer'] + [noise] + [y_batch])

        z_real_decision = z_discriminator(z, training=True)
        z_fake_decision = z_discriminator(recon_z, training=True)
        
        img_real_decision = img_discriminator(x_batch, training=True)
        img_fake_decision1 = img_discriminator(generated_images, training=True)
        img_fake_decision2 = img_discriminator(reconstructed_images, training=True)

        img_dis_loss = img_discriminator_loss(img_real_decision, img_fake_decision1, img_fake_decision2)
        z_dis_loss = z_discriminator_loss(z_real_decision, z_fake_decision)
        
        gen_loss = generator_loss(img_fake_decision1) + generator_loss(img_fake_decision2)
        enc_loss = encoder_loss(z_fake_decision)

    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder.trainable_variables)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_img_discriminator = img_dis_tape.gradient(img_dis_loss, img_discriminator.trainable_variables)
    gradients_of_z_discriminator = z_dis_tape.gradient(z_dis_loss, z_discriminator.trainable_variables)

    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    img_discriminator_optimizer.apply_gradients(zip(gradients_of_img_discriminator, img_discriminator.trainable_variables))
    z_discriminator_optimizer.apply_gradients(zip(gradients_of_z_discriminator, z_discriminator.trainable_variables))
    
    return reconstructed_images, enc_loss, gen_loss, z_dis_loss, img_dis_loss
#%%
step = 0
progress_bar = tqdm(range(PARAMS['epochs']))
progress_bar.set_description('iteration {}/{} | current loss ?'.format(step, PARAMS['epochs']))
#%%
def generate_and_save_images(images, epochs):
    fig = plt.figure(figsize=(4,4))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i, :, :, 0] * 127.5 + 127.5)
        plt.axis('off')

    plt.savefig('./assets/image_at_epoch_{:04d}.png'.format(epochs))
    plt.show()
#%%
'''training'''
# loss_history = []
for _ in progress_bar:
    x_batch, y_batch = next(iter(train_dataset))
    step += 1
    
    reconstructed_images, enc_loss, gen_loss, z_dis_loss, img_dis_loss = train_one_step(x_batch, y_batch)
    # loss_history.append(current_loss.numpy())
    
    # progress_bar.set_description('setting: {} epochs:{} lr:{} dim:{} T:{} sigma:{} to {} | iteration {}/{} | current loss {:.3f}'.format(
    #     PARAMS['data'], PARAMS['epochs'], PARAMS['learning_rate'], PARAMS['embedding_dim'], PARAMS['T'], PARAMS['sigma_0'], PARAMS['sigma_T'], 
    #     step, PARAMS['epochs'], current_loss
    # ))

    if step % 10 == 0:
        generate_and_save_images(reconstructed_images, step)

    if step == PARAMS['epochs']: break
#%%