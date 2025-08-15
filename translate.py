"""
Script to translate a single image using a trained Pix2Pix generator model.
"""

import os
import argparse
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from matplotlib import pyplot


def load_image(filename, size=(256, 256)):
    """
    Load and preprocess a single image.

    Args:
        filename (str): The path to the image file.
        size (tuple, optional): The target size to resize the image to. Defaults to (256, 256).

    Returns:
        numpy.ndarray: The preprocessed image, ready for the generator model.
    """
    pixels = load_img(filename, target_size=size)
    pixels = img_to_array(pixels)
    # Scale from [0,255] to [-1,1]
    pixels = (pixels - 127.5) / 127.5
    # Add a batch dimension
    pixels = expand_dims(pixels, 0)
    return pixels


def main():
    """
    Main function to parse arguments and run the image translation.
    """
    parser = argparse.ArgumentParser(description='Translate an image using a Pix2Pix model.')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained generator model (.h5 file).')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image to translate.')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the translated image.')
    args = parser.parse_args()

    # Load the model
    model = load_model(args.model_path)

    # Load and preprocess the source image
    src_image = load_image(args.image_path)
    print('Loaded source image:', src_image.shape)

    # Generate the translated image
    gen_image = model.predict(src_image)
    # Scale from [-1,1] to [0,1]
    gen_image = (gen_image + 1) / 2.0

    # Save the output image
    pyplot.imsave(args.output_path, gen_image[0])
    print(f'Translated image saved to: {args.output_path}')

    # Optionally, display the image
    # pyplot.imshow(gen_image[0])
    # pyplot.axis('off')
    # pyplot.show()


if __name__ == '__main__':
    main()
