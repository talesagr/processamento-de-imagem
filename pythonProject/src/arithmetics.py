import numpy as np
from PIL import Image


class ArithmeticsOperations:

    def sum_images(self,image1, image2):
        # Obtém as dimensões mínimas
        width = min(image1.width, image2.width)
        height = min(image1.height, image2.height)

        # Redimensiona as imagens para as dimensões mínimas
        image1 = image1.resize((width, height))
        image2 = image2.resize((width, height))

        # Converte para arrays NumPy
        img1_array = np.array(image1, dtype=np.int32)
        img2_array = np.array(image2, dtype=np.int32)

        # Garante que ambas as imagens tenham 3 canais RGB
        if img1_array.ndim == 2:  # Imagem em escala de cinza
            img1_array = np.stack((img1_array,) * 3, axis=-1)
        if img2_array.ndim == 2:
            img2_array = np.stack((img2_array,) * 3, axis=-1)

        # Inicializa o array para o resultado
        sum_img = np.zeros_like(img1_array, dtype=np.int32)

        # Soma pixel a pixel
        for y in range(height):
            for x in range(width):
                # Soma cada canal RGB
                r = min(img1_array[y, x, 0] + img2_array[y, x, 0], 255)
                g = min(img1_array[y, x, 1] + img2_array[y, x, 1], 255)
                b = min(img1_array[y, x, 2] + img2_array[y, x, 2], 255)

                # Define o pixel resultante
                sum_img[y, x] = (r, g, b)

        # Converte para uint8 para garantir compatibilidade com imagens
        sum_img = np.clip(sum_img, 0, 255).astype(np.uint8)

        return sum_img

    def subtract_images(self, image1, image2):
        # Obtém as dimensões mínimas
        width = min(image1.width, image2.width)
        height = min(image1.height, image2.height)

        # Redimensiona as imagens para as dimensões mínimas
        image1 = image1.resize((width, height))
        image2 = image2.resize((width, height))

        # Converte para arrays NumPy
        img1_array = np.array(image1, dtype=np.int32)
        img2_array = np.array(image2, dtype=np.int32)

        # Garante que ambas as imagens tenham 3 canais RGB
        if img1_array.ndim == 2:
            img1_array = np.stack((img1_array,) * 3, axis=-1)
        if img2_array.ndim == 2:
            img2_array = np.stack((img2_array,) * 3, axis=-1)

        # Inicializa o array para o resultado
        subtract_img = np.zeros_like(img1_array, dtype=np.int32)

        # Subtrai pixel a pixel
        for y in range(height):
            for x in range(width):
                r = max(img1_array[y, x, 0] - img2_array[y, x, 0], 0)
                g = max(img1_array[y, x, 1] - img2_array[y, x, 1], 0)
                b = max(img1_array[y, x, 2] - img2_array[y, x, 2], 0)

                subtract_img[y, x] = (r, g, b)

        subtract_img = np.clip(subtract_img, 0, 255).astype(np.uint8)
        return subtract_img

    def multiply_images(self, image1, image2):
        # Obtém as dimensões mínimas
        width = min(image1.width, image2.width)
        height = min(image1.height, image2.height)

        # Redimensiona as imagens para as dimensões mínimas
        image1 = image1.resize((width, height))
        image2 = image2.resize((width, height))

        # Converte para arrays NumPy
        img1_array = np.array(image1, dtype=np.float32) / 255.0
        img2_array = np.array(image2, dtype=np.float32) / 255.0

        # Inicializa a matriz de resultado
        multiply_img = np.zeros_like(img1_array)

        # Percorre cada pixel
        for y in range(height):
            for x in range(width):
                # Multiplica os valores de cada canal RGB
                r = img1_array[y, x, 0] * img2_array[y, x, 0]
                g = img1_array[y, x, 1] * img2_array[y, x, 1]
                b = img1_array[y, x, 2] * img2_array[y, x, 2]

                # Define os valores no resultado
                multiply_img[y, x] = (r, g, b)

        # Escala de volta para o intervalo [0, 255]
        multiply_img = (multiply_img * 255).astype(np.uint8)

        return multiply_img

    def divide_images(self, image1, image2):
        # Obtém as dimensões mínimas
        width = min(image1.width, image2.width)
        height = min(image1.height, image2.height)

        # Redimensiona as imagens para as dimensões mínimas
        image1 = image1.resize((width, height))
        image2 = image2.resize((width, height))

        # Converte para arrays NumPy
        img1_array = np.array(image1, dtype=np.float32)
        img2_array = np.array(image2, dtype=np.float32)

        # Inicializa o array para o resultado
        divide_img = np.zeros_like(img1_array)

        # Realiza a divisão pixel a pixel
        for y in range(height):
            for x in range(width):
                r = img1_array[y, x, 0] / img2_array[y, x, 0] if img2_array[y, x, 0] != 0 else 0
                g = img1_array[y, x, 1] / img2_array[y, x, 1] if img2_array[y, x, 1] != 0 else 0
                b = img1_array[y, x, 2] / img2_array[y, x, 2] if img2_array[y, x, 2] != 0 else 0

                divide_img[y, x] = (r, g, b)

        # Escala para o intervalo [0, 255]
        divide_img = (divide_img / np.max(divide_img) * 255) if np.max(divide_img) > 0 else divide_img
        divide_img = np.clip(divide_img, 0, 255).astype(np.uint8)

        return divide_img

    def media_images(self, image1, image2):
        # Obtém as dimensões mínimas
        width = min(image1.width, image2.width)
        height = min(image1.height, image2.height)

        # Redimensiona as imagens para as dimensões mínimas
        image1 = image1.resize((width, height))
        image2 = image2.resize((width, height))

        # Converte para arrays NumPy
        img1_array = np.array(image1, dtype=np.int32)
        img2_array = np.array(image2, dtype=np.int32)

        # Inicializa o array para o resultado
        medium_img = np.zeros_like(img1_array)

        # Calcula a média pixel a pixel
        for y in range(height):
            for x in range(width):
                r = (img1_array[y, x, 0] + img2_array[y, x, 0]) // 2
                g = (img1_array[y, x, 1] + img2_array[y, x, 1]) // 2
                b = (img1_array[y, x, 2] + img2_array[y, x, 2]) // 2

                medium_img[y, x] = (r, g, b)

        medium_img = np.clip(medium_img, 0, 255).astype(np.uint8)
        return medium_img

    def mediana_images(self, image1, image2):
        # Obtém as dimensões mínimas
        width = min(image1.width, image2.width)
        height = min(image1.height, image2.height)

        # Redimensiona as imagens para as dimensões mínimas
        image1 = image1.resize((width, height))
        image2 = image2.resize((width, height))

        # Converte para arrays NumPy
        img1_array = np.array(image1, dtype=np.int32)
        img2_array = np.array(image2, dtype=np.int32)

        # Inicializa o array para o resultado
        median_img = np.zeros_like(img1_array, dtype=np.int32)

        # Calcula a mediana pixel a pixel
        for y in range(height):
            for x in range(width):
                r_values = [img1_array[y, x, 0], img2_array[y, x, 0]]
                g_values = [img1_array[y, x, 1], img2_array[y, x, 1]]
                b_values = [img1_array[y, x, 2], img2_array[y, x, 2]]

                #Ordena os valores em forma ASC, 1,2,3,4....
                r_values.sort()
                g_values.sort()
                b_values.sort()

                r = r_values[len(r_values) // 2]
                g = g_values[len(g_values) // 2]
                b = b_values[len(b_values) // 2]

                median_img[y, x] = (r, g, b)

        median_img = np.clip(median_img, 0, 255).astype(np.uint8)

        return median_img


    def blend_images(self, img1, img2, alpha=0.7):
        #MUDAR O ALPHA AQUI QUANDO QUISER RESULTADOS DIFERENTES

        img1_array = np.array(img1, dtype=np.float32)
        img2_array = np.array(img2, dtype=np.float32)

        blended_img = np.zeros_like(img1_array, dtype=np.float32)

        if img1_array.ndim == 3:
            height, width, channels = img1_array.shape

            for y in range(height):# blended = alpha * pixel1 + (1 - alpha) * pixel2::: quando alpha = 0, o resultado eh identico a segunda imagm
                for x in range(width):#blended_pixel[R->G->B]=alpha*pixel1[R->G->B]+(1−alpha)*pixel2[R->G->B]
                    for c in range(channels):
                        blended_img[y, x, c] = alpha * img1_array[y, x, c] + (1 - alpha) * img2_array[y, x, c]
        else:
            height, width = img1_array.shape

            for y in range(height):
                for x in range(width):
                    blended_img[y, x] = alpha * img1_array[y, x] + (1 - alpha) * img2_array[y, x]

        return Image.fromarray(np.clip(blended_img, 0, 255).astype(np.uint8))