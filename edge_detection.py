import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import math


def load_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error Loading image: {e}")
        return ValueError("Image not found or invalid format")
    return img

def show_image(image, title="Image"):
    
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_to_gray(image :np.ndarray, check : bool = False):
    if check:
        if len(image.shape) != 3:
            # print("Image is Already in gray scale format")
            return 1
        if len(image.shape) == 3:
            # print("Image is not in gray scale")
            return 0
    h, w, ch = image.shape
    if ch == 3:
        gray_image = np.zeros((h, w), image.dtype)
        r_weight, g_weight, b_weight = 0.299, 0.587, 0.114
        for i in range(h):
            for j in range(w):
                r, g, b = image[i][j]
                gray = r_weight * r + g_weight * g + b_weight * b
                gray_image[i][j] = round(gray)
        return gray_image.astype(image.dtype)
    

def guassian_blur(image):
    # ensure image is in gray scale
    if convert_to_gray(image, True) == 0:
        print("Convert the image to gray scale first")
        sys.exit(0)
    else:
        kernel = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        image = convolve(image, kernel)
        return image


def convolve(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape

    pad_h , pad_w = kh // 2, kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    result = np.zeros(padded.shape, padded.dtype)

    for i in range(h):
        for j in range(w):
            result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
    return result.astype(np.uint8)


def get_sobel(*args):

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    if len(args) == 1:
        if args[0] == 'x':
            return kx
        elif args[0] == 'y':
            return ky
    elif len(args) == 2:
        return (kx, ky)
    else:
        raise ValueError("Entered more than 2 dimensions")


def calc_magnitude(Ix, Iy):
    G = np.hypot(Ix, Iy)
    G = (G / G.max() * 255).astype(np.uint8)
    return G


def program(image_path):
    image = load_image(image_path)
    show_image(image, "original")

    gray_image = convert_to_gray(image)

    blur_image = guassian_blur(gray_image)

    kx, ky = get_sobel('x', 'y')

    Ix = convolve(blur_image, kx)
    Iy = convolve(blur_image, ky)

    G = calc_magnitude(Ix, Iy)

    show_image(G, "Result")


program("images/test-3.jpg")