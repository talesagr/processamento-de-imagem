from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
import matplotlib.pyplot as plt

from arithmetics import ArithmeticsOperations
from logical import LogicalOperations
from gaussian_calcs import GaussianCalcs
from edge_filter import EdgeFilter


class ImageProcessor:

    def __init__(self):
        self.gaussian_calcs = GaussianCalcs()
        self.edge_filter = EdgeFilter()

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
        elif operation == "blending":
            result_image = arithmetics.blend_images(img1, img2)


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

    def convert_to_grayscale(self, image):
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


    def convert_to_binary(self,image):
        threshold = 128
        if image.mode != "L":
            grayscale_image = self.convert_to_grayscale(image)
        else:
            grayscale_image = image

        img_array = np.array(grayscale_image, dtype=np.uint8)

        binary_img = np.zeros_like(img_array, dtype=np.uint8)

        height, width = img_array.shape
        for y in range(height):
            for x in range(width):
                #se o pixel for maior que 128 entao ele eh considerado 255, caso contrario = 0
                binary_img[y, x] = 255 if img_array[y, x] > threshold else 0

        # Converte o array resultante para uma imagem PIL
        return Image.fromarray(binary_img)


    @staticmethod
    def convert_to_negative(image):
        img_array = np.array(image, dtype=np.uint8)

        negative_img = np.zeros_like(img_array, dtype=np.uint8)

        if img_array.ndim == 3:
            height, width, channels = img_array.shape

            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        negative_img[y, x, c] = 255 - img_array[y, x, c]
        else:
            height, width = img_array.shape

            for y in range(height):
                for x in range(width):
                    negative_img[y, x] = 255 - img_array[y, x]

        return Image.fromarray(negative_img)

    def apply_gaussian_filter(self,image):
        img_array = np.array(image, dtype=np.float32)

        # Tamanho do kernel
        kernel_size = 5 #sempre um valor impar
        half_size = kernel_size // 2

        # Criação do kernel gaussiano
        GKernel = self.gaussian_kernel(kernel_size)

        # Aplicação do filtro gaussiano
        height, width, channels = img_array.shape
        filtered_image = np.zeros_like(img_array)

        for c in range(channels):
            for x in range(half_size, width - half_size):
                for y in range(half_size, height - half_size):
                    pixel_value = 0

                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            pixel_value += GKernel[i, j] * img_array[y - half_size + i, x - half_size + j, c]

                    filtered_image[y, x, c] = pixel_value

        # Converte o resultado para uint8 e cria uma imagem PIL
        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

        return Image.fromarray(filtered_image)



    def gaussian_kernel(self, kernel_size):
        sigma = 0.6

        GKernel = [[0.0 for _ in range(kernel_size)] for _ in range(kernel_size)]
        sum_kernel = 0.0
        pi = 3.141592653589793

        """
        Calculo do valor do coeficiente, a formula sendo 1 / 2*pi*sigma²
        """
        coefficient = 1.0 / (2.0 * pi * sigma * sigma)
        # Calcula o ponto central do kernel
        half_size = kernel_size // 2

        # Calcula os valores do kernel
        """
        O kernel eh uma matriz NxN (no caso, 5x5)
        definindo o half_size como nosso range, a gente garante que a nossa matriz seja preenchida com:
        o valor NEGATIVO da metade do tamanho da matriz até o valor POSITIVO da metado do tamanho da matriz
        algo como:
        -2 -1  0 -1 -2
        """
        for x in range(-half_size, half_size + 1):
            for y in range(-half_size, half_size + 1):
                """
                formula do calculo do exponencial x²+y² / 2*sigma²
                por isso o sigma controla a suavidade do filtro, quanto maior o valor, mais suave
                """
                exponent = -(x ** 2 + y ** 2) / (2.0 * sigma * sigma)
                """
                Aqui a gente multiplica o coeficiente pelo exponencial do expoente
                """
                value = coefficient * self.gaussian_calcs.exp_manual(exponent)

                """
                estamos preenchendo a matriz do kernel  nas posicoes que estamos percorrendo 
                  com o resultado dos calcumos efetuados acima 
                """
                GKernel[x + half_size][y + half_size] = value

                """
                utilizamos esse valor abaixo para garanir que a soma de tudo seja 1
                """
                sum_kernel += value

        # Normaliza o kernel para garantir que a soma dos elementos seja 1
        for i in range(kernel_size):
            for j in range(kernel_size):
                GKernel[i][j] /= sum_kernel

        return np.array(GKernel, dtype=np.float32)


    def edge_detection(self,image, method="sobel"):
        img_array = np.array(image, dtype=np.int32)

        if method == "sobel":
            return self.edge_filter.sobel(img_array)
        elif method == "laplacian":
            return self.edge_filter.laplacian(img_array)
        elif method == "prewitt":
            return self.edge_filter.prewitt(img_array)


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

    def increase_bright(self, image):
        img_array = np.array(image, dtype=np.int32)

        if img_array.ndim == 3:
            height, width, channels = img_array.shape
            bright_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        bright_img[y, x, c] = min(img_array[y, x, c] + 50, 255)
        else:
            height, width = img_array.shape
            bright_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    bright_img[y, x] = min(img_array[y, x] + 50, 255)

        return Image.fromarray(np.clip(bright_img, 0, 255).astype(np.uint8))

    def decrease_bright(self, image):
        img_array = np.array(image, dtype=np.int32)

        if img_array.ndim == 3:
            height, width, channels = img_array.shape
            bright_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        bright_img[y, x, c] = max(img_array[y, x, c] - 50, 0)
        else:
            height, width = img_array.shape
            bright_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    bright_img[y, x] = max(img_array[y, x] - 50, 0)

        return Image.fromarray(np.clip(bright_img, 0, 255).astype(np.uint8))

    def increase_contrast(self, image):#40%
        img_array = np.array(image, dtype=np.int32)

        midpoint = 128

        if img_array.ndim == 3:
            height, width, channels = img_array.shape
            contrast_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    for c in range(channels):# novo_valor = fator × ( valor_pixel − media) + media
                        contrast_img[y, x, c] = int(1.4 * (img_array[y, x, c] - midpoint) + midpoint)
        else:
            height, width = img_array.shape
            contrast_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    contrast_img[y, x] = int(1.4 * (img_array[y, x] - midpoint) + midpoint)

        return Image.fromarray(np.clip(contrast_img, 0, 255).astype(np.uint8))

    def decrease_contrast(self, image):#40%
        img_array = np.array(image, dtype=np.int32)

        midpoint = 128

        if img_array.ndim == 3:
            height, width, channels = img_array.shape
            contrast_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    for c in range(channels):
                        contrast_img[y, x, c] = int(0.6 * (img_array[y, x, c] - midpoint) + midpoint)
        else:
            height, width = img_array.shape
            contrast_img = np.zeros_like(img_array)

            for y in range(height):
                for x in range(width):
                    contrast_img[y, x] = int(0.6 * (img_array[y, x] - midpoint) + midpoint)

        return Image.fromarray(np.clip(contrast_img, 0, 255).astype(np.uint8))

    def plot_histograms(self, image, equalized_image):
        original_array = np.array(image)
        equalized_array = np.array(equalized_image)

        plt.figure(figsize=(12, 6))

        # Histograma da imagem original
        plt.subplot(1, 2, 1)
        plt.hist(original_array.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
        plt.title("Histograma Original")
        plt.xlabel("Intensidade de Cinza")
        plt.ylabel("Frequência")

        # Histograma da imagem equalizada
        plt.subplot(1, 2, 2)
        plt.hist(equalized_array.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
        plt.title("Histograma Equalizado")
        plt.xlabel("Intensidade de Cinza")
        plt.ylabel("Frequência")

        # Exibe os gráficos
        plt.tight_layout()
        plt.show()


    def spin_X(self, image):
        img_array = np.array(image)
        height, width, channels = img_array.shape

        flipped_img = np.zeros_like(img_array)

        for y in range(height):
            for x in range(width):
                flipped_img[y, x] = img_array[height - y - 1, x]

        return Image.fromarray(flipped_img)

    def spin_Y(self, image):
        img_array = np.array(image)
        height, width, channels = img_array.shape

        flipped_img = np.zeros_like(img_array)

        for y in range(height):
            for x in range(width):
                flipped_img[y, x] = img_array[y, width - x - 1]

        return Image.fromarray(flipped_img)

    def apply_order(self, image, order):
        img_array = np.array(image, dtype=np.float32)

        # Tamanho do kernel
        kernel_size = 5  # sempre um valor impar
        half_size = kernel_size // 2

        # Criação do kernel gaussiano
        GKernel = self.gaussian_kernel(kernel_size)

        # Aplicação do filtro gaussiano
        height, width, channels = img_array.shape
        filtered_image = np.zeros_like(img_array)

        for x in range(half_size, width - half_size):
            for y in range(half_size, height - half_size):
                for c in range(channels):  # Processa cada canal separadamente
                    neighbors = []
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            neighbors.append(
                                img_array[y - half_size + i, x - half_size + j, c]
                            )

                    # Ordena os valores da vizinhança
                    neighbors.sort()

                    # Escolhe o valor com base no tipo de filtro
                    if order == "med":
                        value = neighbors[len(neighbors) // 2]
                    elif order == "min":
                        value = neighbors[0]
                    elif order == "max":
                        value = neighbors[-1]
                    else:
                        raise ValueError(f"Filtro de ordem desconhecido: {order}")

                    # Define o valor do pixel na imagem filtrada
                    filtered_image[y, x, c] = value

        filtered_image = np.clip(filtered_image, 0, 255).astype(np.uint8)

        return Image.fromarray(filtered_image)