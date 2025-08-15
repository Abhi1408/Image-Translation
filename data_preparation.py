"""
Script to prepare the dataset for the Pix2Pix model.

This script takes a directory of combined satellite and map images,
splits them, and saves them as a compressed NumPy array (.npz) file.
The input images are expected to be in a format where the satellite image
is on the left and the map image is on the right.
"""

import os
import argparse
from numpy import asarray
from numpy import savez_compressed
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def load_images(path, size=(256, 512)):
    """
    Load and split all images in a directory into source (satellite) and target (map) images.

    Args:
        path (str): The path to the directory containing the images.
        size (tuple, optional): The target size to resize the images to.
                                Defaults to (256, 512).

    Returns:
        list: A list containing two NumPy arrays: one for source images and one for target images.
    """
    src_list, tar_list = list(), list()
    # Enumerate filenames in directory, ignoring subdirectories
    for filename in os.listdir(path):
        # Construct full path to image
        image_path = os.path.join(path, filename)
        # Skip directories
        if not os.path.isfile(image_path):
            continue
        # Load the image
        pixels = load_img(image_path, target_size=size)
        # Convert to an array
        pixels = img_to_array(pixels)
        # Split into satellite and map
        sat_img, map_img = pixels[:, :256], pixels[:, 256:]
        src_list.append(sat_img)
        tar_list.append(map_img)
    return [asarray(src_list), asarray(tar_list)]


def main():
    """
    Main function to parse arguments and run the data preparation process.
    """
    parser = argparse.ArgumentParser(description='Prepare dataset for Pix2Pix model.')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to the directory containing the training images.')
    parser.add_argument('--output_file', type=str, default='maps_256.npz',
                        help='Name of the output .npz file.')
    args = parser.parse_args()

    # Load dataset
    [src_images, tar_images] = load_images(args.input_dir)
    print('Loaded: ', src_images.shape, tar_images.shape)

    # Save as compressed numpy array
    savez_compressed(args.output_file, src_images, tar_images)
    print('Saved dataset: ', args.output_file)


if __name__ == '__main__':
    main()
