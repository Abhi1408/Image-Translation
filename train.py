"""
Main script for training the Pix2Pix model for image-to-image translation.
"""

import os
import argparse
from numpy import load, zeros, ones
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization
from matplotlib import pyplot


def define_discriminator(image_shape):
    """
    Define the discriminator model for the Pix2Pix GAN.

    The discriminator is a PatchGAN that classifies 70x70 patches of the image as real or fake.

    Args:
        image_shape (tuple): The shape of the input images (height, width, channels).

    Returns:
        tensorflow.keras.models.Model: The compiled discriminator model.
    """
    init = RandomNormal(stddev=0.02)
    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)
    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([in_src_image, in_target_image], patch_out)
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def define_encoder_block(layer_in, n_filters, batchnorm=True):
    """
    Define a block in the U-Net encoder.

    Args:
        layer_in (tensorflow.keras.layers.Layer): The input layer.
        n_filters (int): The number of filters for the convolutional layer.
        batchnorm (bool, optional): Whether to use batch normalization. Defaults to True.

    Returns:
        tensorflow.keras.layers.Layer: The output layer of the encoder block.
    """
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g


def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    """
    Define a block in the U-Net decoder.

    Args:
        layer_in (tensorflow.keras.layers.Layer): The input layer from the previous decoder block.
        skip_in (tensorflow.keras.layers.Layer): The skip connection layer from the corresponding encoder block.
        n_filters (int): The number of filters for the transposed convolutional layer.
        dropout (bool, optional): Whether to use dropout. Defaults to True.

    Returns:
        tensorflow.keras.layers.Layer: The output layer of the decoder block.
    """
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    if dropout:
        g = Dropout(0.5)(g, training=True)
    g = Concatenate()([g, skip_in])
    g = Activation('relu')(g)
    return g


def define_generator(image_shape=(256, 256, 3)):
    """
    Define the U-Net generator model.

    Args:
        image_shape (tuple, optional): The shape of the input images. Defaults to (256, 256, 3).

    Returns:
        tensorflow.keras.models.Model: The generator model.
    """
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    # Encoder
    e1 = define_encoder_block(in_image, 64, batchnorm=False)
    e2 = define_encoder_block(e1, 128)
    e3 = define_encoder_block(e2, 256)
    e4 = define_encoder_block(e3, 512)
    e5 = define_encoder_block(e4, 512)
    e6 = define_encoder_block(e5, 512)
    e7 = define_encoder_block(e6, 512)

    # Bottleneck
    b = Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)

    # Decoder
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)

    # Output
    g = Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model


def define_gan(g_model, d_model, image_shape):
    """
    Define the combined GAN model for training the generator.

    Args:
        g_model (tensorflow.keras.models.Model): The generator model.
        d_model (tensorflow.keras.models.Model): The discriminator model.
        image_shape (tuple): The shape of the input images.

    Returns:
        tensorflow.keras.models.Model: The compiled GAN model.
    """
    d_model.trainable = False
    in_src = Input(shape=image_shape)
    gen_out = g_model(in_src)
    dis_out = d_model([in_src, gen_out])
    model = Model(in_src, [dis_out, gen_out])
    opt = Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
    return model


def load_real_samples(filename, direction='sat2map'):
    """
    Load and preprocess the dataset from a .npz file.

    Args:
        filename (str): The path to the .npz file.
        direction (str, optional): The translation direction, either 'sat2map' or 'map2sat'.
                                   Defaults to 'sat2map'.

    Returns:
        list: A list containing two NumPy arrays: one for source images and one for target images.
    """
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    # Scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    if direction == 'map2sat':
        return [X2, X1]
    return [X1, X2]


def generate_real_samples(dataset, n_samples, patch_shape):
    """
    Select a batch of random real samples from the dataset.

    Args:
        dataset (list): A list of two NumPy arrays (source and target images).
        n_samples (int): The number of samples to select.
        patch_shape (int): The shape of the discriminator's output patch.

    Returns:
        tuple: A tuple containing the selected images and the 'real' class labels.
    """
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y


def generate_fake_samples(g_model, samples, patch_shape):
    """
    Generate a batch of fake samples using the generator.

    Args:
        g_model (tensorflow.keras.models.Model): The generator model.
        samples (numpy.ndarray): The input samples for the generator.
        patch_shape (int): The shape of the discriminator's output patch.

    Returns:
        tuple: A tuple containing the generated images and the 'fake' class labels.
    """
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y


def summarize_performance(step, g_model, dataset, direction, n_samples=3):
    """
    Summarize the performance of the generator by saving generated images and the model.

    Args:
        step (int): The current training step.
        g_model (tensorflow.keras.models.Model): The generator model.
        dataset (list): The dataset to sample from for generating images.
        direction (str): The translation direction, for naming output files.
        n_samples (int, optional): The number of samples to plot. Defaults to 3.
    """
    [X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
    X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)

    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0

    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        pyplot.imshow(X_fakeB[i])
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples * 2 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realB[i])

    # Save plot to file
    plot_filename = f"plot_{direction}_{step+1:06d}.png"
    pyplot.savefig(os.path.join('images', plot_filename))
    pyplot.close()

    # Save the generator model
    model_filename = f"model_{direction}_{step+1:06d}.h5"
    g_model.save(os.path.join('checkpoints', model_filename))

    print(f'>Saved: {plot_filename} and {model_filename}')


def train(d_model, g_model, gan_model, dataset, direction, n_epochs=100, n_batch=1):
    """
    Train the Pix2Pix model.

    Args:
        d_model (tensorflow.keras.models.Model): The discriminator model.
        g_model (tensorflow.keras.models.Model): The generator model.
        gan_model (tensorflow.keras.models.Model): The combined GAN model.
        dataset (list): The training dataset.
        direction (str): The translation direction.
        n_epochs (int, optional): The number of training epochs. Defaults to 100.
        n_batch (int, optional): The batch size. Defaults to 1.
    """
    n_patch = d_model.output_shape[1]
    trainA, _ = dataset
    bat_per_epo = int(len(trainA) / n_batch)
    n_steps = bat_per_epo * n_epochs

    for i in range(n_steps):
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
        print(f'>{i+1}, d1[{d_loss1:.3f}] d2[{d_loss2:.3f}] g[{g_loss:.3f}]')

        if (i + 1) % (bat_per_epo * 10) == 0:
            summarize_performance(i, g_model, dataset, direction)


def main():
    """
    Main function to parse arguments and run the training process.
    """
    parser = argparse.ArgumentParser(description='Train Pix2Pix model.')
    parser.add_argument('--direction', type=str, default='sat2map', choices=['sat2map', 'map2sat'],
                        help='Translation direction: sat2map or map2sat.')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training.')
    parser.add_argument('--dataset_path', type=str, default='maps_256.npz',
                        help='Path to the .npz dataset file.')
    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('images', exist_ok=True)

    dataset = load_real_samples(args.dataset_path, args.direction)
    print(f'Loaded dataset: {dataset[0].shape}, {dataset[1].shape}')

    image_shape = dataset[0].shape[1:]
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)

    train(d_model, g_model, gan_model, dataset, args.direction, args.n_epochs, args.batch_size)


if __name__ == '__main__':
    main()
