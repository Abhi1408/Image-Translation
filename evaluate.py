"""
Script to evaluate a trained Pix2Pix model using Grad-CAM or LIME for explainability.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
import cv2
from numpy import load, vstack
from numpy.random import randint
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model, Model
from lime import lime_image
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import mark_boundaries


# --- Data Loading ---
def load_real_samples(filename):
    """Load and preprocess samples from a .npz file."""
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]


# --- Plotting ---
def plot_evaluation(src_img, gen_img, tar_img, explanation_img, method_name, explanation_text):
    """
    Plot the source, generated, target, and explanation images.
    """
    images = vstack((src_img, gen_img, tar_img))
    images = (images + 1) / 2.0
    titles = ['Satellite (Input)', 'Generated Map', 'Actual Map', f'{method_name} Explanation']

    plt.figure(figsize=(16, 8))

    # Plot the three main images
    for i in range(3):
        plt.subplot(2, 4, 1 + i)
        plt.axis('off')
        plt.imshow(images[i])
        plt.title(titles[i])

    # Plot the explanation image
    plt.subplot(2, 4, 4)
    plt.axis('off')
    plt.imshow(explanation_img)
    plt.title(titles[3])

    # Add the explanation text below
    plt.subplot(2, 1, 2)
    plt.axis('off')
    plt.text(0.0, 0.5, explanation_text, wrap=True, fontsize=12, va='center')

    plt.tight_layout()
    plt.show()


# --- Grad-CAM Implementation ---
def compute_gradcam(model, image, layer_name='conv2d_12'):
    """
    Compute Grad-CAM heatmap for a given image and model.
    """
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.convert_to_tensor(image))
        loss = tf.reduce_mean(predictions)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)

    # Resize and apply colormap
    heatmap = cv2.resize(heatmap, (256, 256))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap
    src_img_disp = np.uint8(((image[0] + 1) / 2.0) * 255)
    overlayed = cv2.addWeighted(src_img_disp, 0.6, heatmap, 0.4, 0)
    return overlayed


# --- LIME Implementation ---
def compute_ssim(img1, img2):
    """Compute Structural Similarity Index (SSIM) between two images."""
    img1_u8 = np.uint8((img1 + 1) * 127.5)
    img2_u8 = np.uint8((img2 + 1) * 127.5)
    return ssim(img1_u8, img2_u8, multichannel=True, data_range=255)


def explain_with_lime(model, src_image, tar_image):
    """
    Generate a LIME explanation for a model's prediction.
    """
    def predict_fn(perturbed_imgs):
        # LIME generates images in a different format, so we need to resize and normalize
        perturbed_imgs_resized = np.array([cv2.resize(img, (256, 256)) for img in perturbed_imgs])
        perturbed_imgs_normalized = (perturbed_imgs_resized / 127.5) - 1.0

        pred_maps = model.predict(perturbed_imgs_normalized)

        # Calculate SSIM score between each generated map and the target map
        scores = np.array([compute_ssim(pred_map, tar_image[0]) for pred_map in pred_maps])
        return scores.reshape(-1, 1)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.uint8((src_image[0] + 1) * 127.5),
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000,
        batch_size=10
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
    )
    return mark_boundaries(temp / 255.0, mask)


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate a Pix2Pix model with Grad-CAM or LIME.')
    parser.add_argument('--method', type=str, required=True, choices=['gradcam', 'lime'],
                        help='Evaluation method: "gradcam" or "lime".')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained generator model (.h5 file).')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the .npz dataset file for sampling images.')
    args = parser.parse_args()

    # Load model and data
    model = load_model(args.model_path)
    [X1, X2] = load_real_samples(args.dataset_path)
    print(f'Loaded dataset with {X1.shape[0]} samples.')

    # Select a random sample
    ix = randint(0, len(X1), 1)
    src_image, tar_image = X1[ix], X2[ix]
    gen_image = model.predict(src_image)

    explanation_img = None
    explanation_text = ""

    if args.method == 'gradcam':
        explanation_img = compute_gradcam(model, src_image)
        explanation_text = (
            "Grad-CAM highlights the areas in the input satellite image that the generator focused on most.\n"
            "🔴 Red/orange areas indicate high activation, suggesting these features (e.g., roads, buildings)\n"
            "were critical for generating the corresponding map regions."
        )
    elif args.method == 'lime':
        explanation_img = explain_with_lime(model, src_image, tar_image)
        explanation_text = (
            "LIME shows the most influential pixels in the input for the final output.\n"
            "The highlighted regions are areas that, when perturbed, had the most significant impact\n"
            "on the similarity (SSIM score) between the generated map and the actual map."
        )

    # Plot the results
    plot_evaluation(src_image, gen_image, tar_image, explanation_img, args.method.upper(), explanation_text)


if __name__ == '__main__':
    main()
