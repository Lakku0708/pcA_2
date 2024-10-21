import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage import io, color

# Option 1: Load a grayscale image from a URL (use another URL or a valid one)
try:
    # Try loading the image from a URL
    image = io.imread('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png')
    print("Image loaded from URL.")
except Exception as e:
    print(f"Error loading image from URL: {e}")
    # Option 2: Load a grayscale image from a local file (replace 'path_to_image' with your file path)
    image = io.imread('path_to_image.png')  # Replace 'path_to_image.png' with your image path
    print("Image loaded from local file.")

# Convert to grayscale if the image is not already in grayscale
grayscale_image = color.rgb2gray(image)

# Display the original image
plt.imshow(grayscale_image, cmap='gray')
plt.title('Original Image')
plt.show()

# Reshape the image to a 2D matrix where each row is a pixel and each column is a feature
image_shape = grayscale_image.shape
image_reshaped = grayscale_image.reshape(image_shape[0], -1)

# Function to apply PCA and reconstruct the image
def apply_pca(image, n_components):
    pca = PCA(n_components=n_components)
    pca_transformed = pca.fit_transform(image)
    reconstructed = pca.inverse_transform(pca_transformed)
    return reconstructed

# Apply PCA with different numbers of components
components = [5, 20, 50, 100]
reconstructed_images = [apply_pca(image_reshaped, n) for n in components]

# Plot the original image and the reconstructed images
plt.figure(figsize=(10, 8))

# Display the original image
plt.subplot(2, 3, 1)
plt.imshow(grayscale_image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

# Display the reconstructed images with different number of components
for i, (n_comp, recon_img) in enumerate(zip(components, reconstructed_images)):
    plt.subplot(2, 3, i + 2)
    plt.imshow(recon_img.reshape(image_shape), cmap='gray')
    plt.title(f'{n_comp} Components')
    plt.axis('off')

plt.tight_layout()
plt.show()
