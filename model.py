import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np

def build_generator():
    model = Sequential()
    model.add(Dense(7*7*256, input_dim=100))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 256)))
    model.add(Conv2DTranspose(128, (5,5), strides=(1,1), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(64, (5,5), strides=(2,2), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', activation='tanh'))
    return model

def generate_image(generator):
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)
    generated_image = (generated_image + 1) / 2.0  # Rescale 0-1
    return generated_image[0, :, :, 0]

generator = build_generator()
generator.load_weights('generator.weights.h5')

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    generated_image = generate_image(generator)
    plt.imshow(generated_image, cmap='gray')
    plt.show()
