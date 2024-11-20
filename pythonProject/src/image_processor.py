from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

from arithmetics import ArithmeticsOperations
from logical import LogicalOperations


class ImageProcessor:

    def __init__(self):
        pass

    @staticmethod
    def load_image(image_path):
        img = Image.open(image_path)
        img_array = np.array(img)
        return img_array

    def arithmetic_operation(self, img1, img2, operation):
        img1, img2 = self.check_image_compatibility(img1, img2)

        if img1.size != img2.size:
            raise ValueError("As imagens devem ter o mesmo tamanho")

        arithmetics = ArithmeticsOperations()


        if operation == "sum":
            result_image = arithmetics.sum_images(img1, img2)
        elif operation == "subtract":
            result_image = arithmetics.subtract_images(img1, img2)
        elif operation == "multiply":
            result_image = arithmetics.multiply_images(img1, img2)
        elif operation == "divide":
            result_image = arithmetics.divide_images(img1, img2)
        elif operation == "media":
            result_image = arithmetics.media_images(img1, img2)
        elif operation == "mediana":
            result_image = arithmetics.mediana_images(img1, img2)

        result_image = np.clip(result_image, 0, 255).astype(np.uint8)
        return result_image

    @staticmethod
    def logical_operation(img1, img2, operation):
        if img2 is not None and img1.shape != img2.shape:
            raise ValueError("As imagens devem ter o mesmo tamanho")

        logical = LogicalOperations()

        if operation == "and":
            result_image = logical.and_operation(img1, img2)
        elif operation == "or":
            result_image = logical.or_operation(img1, img2)
        elif operation == "xor":
            result_image = logical.xor_operation(img1, img2)
        elif operation == "not":
            result_image = logical.not_operation(img1)

        return result_image

    @staticmethod
    def convert_to_grayscale(image):
        image_array = np.array(image)

        if image_array.ndim != 3 or image_array.shape[2] != 3:
            raise ValueError("A imagem deve ser uma imagem RGB com 3 canais.")

        grayscale_img = np.zeros((image_array.shape[0], image_array.shape[1]), dtype=np.uint8)

        for i in range(image_array.shape[0]):
            for j in range(image_array.shape[1]):
                r, g, b = image_array[i, j].astype(np.int32)
                gray = int((r + g + b) / 3)
                grayscale_img[i, j] = gray

        return grayscale_img

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


    def check_image_compatibility(self, image1, image2):
        # Converte image1 para PIL.Image se necessário
        if not isinstance(image1, Image.Image):
            if isinstance(image1, np.ndarray):
                image1 = Image.fromarray(image1)
            else:
                raise ValueError("image1 não é compatível. Deve ser um array NumPy ou objeto PIL.Image.")

        # Converte image2 para PIL.Image se necessário
        if not isinstance(image2, Image.Image):
            if isinstance(image2, np.ndarray):
                image2 = Image.fromarray(image2)
            else:
                raise ValueError("image2 não é compatível. Deve ser um array NumPy ou objeto PIL.Image.")

        return image1,image2