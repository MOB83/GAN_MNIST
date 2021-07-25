from keras import initializers
from keras.layers.core import Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential


def build_discrinimator(input_dim, optimizer):
    dis = Sequential()
    dis.add(Dense(1024, input_dim=input_dim, kernel_initializer=initializers.RandomNormal(stddev=0.2)))
    dis.add(LeakyReLU(0.2))
    dis.add(Dropout(0.3))

    dis.add(Dense(512))
    dis.add(LeakyReLU(0.2))
    dis.add(Dropout(0.3))

    dis.add(Dense(256))
    dis.add(LeakyReLU(0.2))
    dis.add(Dropout(0.3))

    dis.add(Dense(1, activation='sigmoid'))
    dis.compile(loss='binary_crossentropy', optimizer=optimizer)

    return dis