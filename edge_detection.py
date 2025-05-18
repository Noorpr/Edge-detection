import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math


def load_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or invalid format")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        print(f"Error Loading image: {e}")
        sys.exit(1)


def show_image(image, title="Image"):
    # Convert back to BGR for cv2.imshow
    if len(image.shape) == 3:
        image_display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_display = image  # Grayscale image
    
    cv2.imshow(title, image_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def is_grayscale(image):
    return len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1)


def convert_to_gray(image):
    if is_grayscale(image):
        return image
    return np.round(image[:,:,0] * 0.299 + image[:,:,1] * 0.587 + image[:,:,2] * 0.114).astype(np.uint8)


def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    result = np.zeros((h, w), dtype=np.float32)
    for i in range(kh):
        for j in range(kw):
            result += padded[i:i+h, j:j+w] * kernel[i, j]
    
    return np.clip(result, 0, 255).astype(np.uint8)


def gaussian_blur(image, kernel_size=3):
    # Ensure image is in grayscale
    if not is_grayscale(image):
        print("Converting image to grayscale first")
        image = convert_to_gray(image)
    
    # Create a proper Gaussian kernel of the specified size
    if kernel_size == 3:
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
    else:
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.outer(kernel, kernel)
    
    blurred = convolve(image, kernel)
    return blurred


def get_sobel(*args):
    """
    Get Sobel filters with specified size.
    
    Args:
        size: Integer, size of the Sobel filter (must be odd)
        *args: 'x' for x-direction filter, 'y' for y-direction filter, or both
    
    Returns:
        Single filter or tuple of filters based on args
    """
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    if len(args) == 1:
        if args[0] == 'x':
            return kx
        elif args[0] == 'y':
            return ky
        else:
            raise ValueError(f"Unknown direction: {args[0]}")
    elif len(args) == 0 or len(args) == 2:
        return (kx, ky)
    else:
        raise ValueError("Invalid number of arguments")


def calc_magnitude(Ix, Iy):
    G = np.hypot(Ix, Iy)
    G = (G / G.max() * 255).astype(np.uint8)
    return G


def edge_detection(image_path, blur_size=3, show_steps=False, save_result=False):
    """
    Perform edge detection using Sobel filters.
    
    Args:
        image_path: Path to the input image
        sobel_size: Size of the Sobel filter (odd number)
        blur_size: Size of the Gaussian blur kernel (odd number)
    
    Returns:
        Edge magnitude image
    """
    # Validate parameters
    if blur_size % 2 == 0:
        raise ValueError("Filter size must be odd numbers")
    
    # Load and display original image
    image = load_image(image_path)
    show_image(image, "Original Image")

    # Convert to grayscale and apply Gaussian blur
    gray_image = convert_to_gray(image)
    blur_image = gaussian_blur(gray_image, blur_size)

    if show_steps:
        show_image(image, "Original Image")
        show_image(gray_image, "Grayscale")
        show_image(blur_image, "Blurred")
    
    # Get Sobel kernels and apply convolution
    kx, ky = get_sobel()
    Ix = convolve(blur_image, kx)
    Iy = convolve(blur_image, ky)

    # Calculate gradient magnitude
    G = calc_magnitude(Ix, Iy)

    # Display results
    show_image(G, "Edge Detection")
    
    if save_result:
        cv2.imwrite("edge_result.jpg", cv2.cvtColor(G, cv2.COLOR_RGB2BGR))
    return G


if __name__ == "__main__":
    edge_detection("images/test-3.jpg", blur_size=3, show_steps=False)