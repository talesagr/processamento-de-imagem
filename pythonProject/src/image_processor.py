from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ImageProcessor:

    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        return img_array

    @staticmethod
    def arithmetic_operation(img1, img2, operation):
        if img1.shape != img2.shape:
            raise ValueError("As imagens devem ter o mesmo tamanho")

        if operation == "sum":
            result_image = img1 + img2
        elif operation == "subtract":
            result_image = img1 - img2
        elif operation == "multiply":
            result_image = img1 * img2
        elif operation == "divide":
            result_image = np.divide(img1, img2, out=np.zeros_like(img1, dtype=float), where=img2 != 0)

        result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        return result_image

    @staticmethod
    def logical_operation(img1, img2, operation):
        if img2 is not None and img1.shape != img2.shape:
            raise ValueError("As imagens devem ter o mesmo tamanho")

        if operation == "and":
            result_image = cv2.bitwise_and(img1, img2)
        elif operation == "or":
            result_image = cv2.bitwise_or(img1, img2)
        elif operation == "xor":
            result_image = cv2.bitwise_xor(img1, img2)
        elif operation == "not":
            result_image = cv2.bitwise_not(img1)

        return result_image

    @staticmethod
    def convert_to_grayscale(image):
        return image.convert("L")

    @staticmethod
    def convert_to_binary(image):
        grayscale_image = image.convert("L")
        binary_image = grayscale_image.point(lambda p: p > 128 and 255)
        return binary_image

    @staticmethod
    def convert_to_negative(image):
        return ImageOps.invert(image)

    @staticmethod
    def apply_gaussian_filter(image):
        return image.filter(ImageFilter.GaussianBlur(radius=2))

    @staticmethod
    def edge_detection(image, method="sobel"):
        gray_image = np.array(image.convert("L"))

        if method == "sobel":
            sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            return Image.fromarray(cv2.magnitude(sobelx, sobely))
        elif method == "laplacian":
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            return Image.fromarray(laplacian)

    @staticmethod
    def equalize_histogram(image):
        if image.mode != "L":
            raise ValueError("A equalização de histograma requer uma imagem em escala de cinza")

        image_array = np.array(image)

        equalized_image = cv2.equalizeHist(image_array)

        return Image.fromarray(equalized_image)