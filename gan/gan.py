import numpy as np

from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model

from tqdm import tqdm

from gan.utils import load_mnist_data, plot_generated_images
from gan.discriminator import build_discrinimator
from gan.generator import build_generator

def build_GAN(random_dim):
    # Define the optimizer used by both networks
    optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

    # Initialise the networks
    dis = build_discrinimator(784, optimizer)
    gen = build_generator(random_dim, optimizer)

    # Setup GAN variables
    dis.trainable = False
    ganInput = Input(shape=(random_dim,))

    # Build the model
    x = gen(ganInput)
    ganOutput = dis(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    return dis, gen, gan

def train_gan(random_dim, generator, discriminator, gan, output_path, epochs=1, batch_size=128):
    x_train, y_train, x_test, y_test = load_mnist_data()
    batch_count = x_train.shape[0] / batch_size

    for e in range(1, epochs+1):
        print('-'*10, 'Epoch %d' % e, '-'*10)
        for _ in tqdm(range(int(batch_count))):
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            image_batch = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            generated_images = generator.predict(noise)

            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2*batch_size)
            y_dis[:batch_size] = 0.9
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)
            noise = np.random.normal(0, 1, size=[batch_size, random_dim])
            y_gen = np.ones(batch_size)
            discriminator.trainable = False
            gan.train_on_batch(noise, y_gen)

        plot_generated_images(output_path, random_dim, e, generator)