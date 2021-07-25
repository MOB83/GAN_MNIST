import numpy as np
import os
from gan.gan import build_GAN, train_gan

# Tensorflow Settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#######
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
#######

RANDOM_SEED = 1000
RANDOM_DIM = 100
OUTPUT_PATH = 'gen_output'

EPOCHS = 25
BATCH_SIZE = 128

np.random.seed(RANDOM_SEED)

dis, gen, gan = build_GAN(RANDOM_DIM)
train_gan(RANDOM_DIM, gen, dis, gan, OUTPUT_PATH, \
    epochs=EPOCHS, batch_size=BATCH_SIZE)

