import tensorflow as tf
from model_def import build_generator
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np


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
