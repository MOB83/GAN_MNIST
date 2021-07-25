from keras import initializers
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

# Generator network
def build_generator(random_dim, optimizer):
    gen = Sequential()
    gen.add(Dense(256, input_dim=random_dim, kernel_initializer=initializers.RandomNormal(stddev=0.2)))
    gen.add(LeakyReLU(0.2))
    gen.add(Dense(512))
    gen.add(LeakyReLU(0.2))
    gen.add(Dense(1024))
    gen.add(LeakyReLU(0.2))
    gen.add(Dense(784, activation='tanh'))
    gen.compile(loss='binary_crossentropy', optimizer=optimizer)

    return gen