import numpy as np
from PIL import Image

class MorphologicsFilters:
    def __init__(self):
        #Da pra mudar o kernel aqui e obter diferentes operacoes
        self.kernel = np.array([
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.uint8)
        self.kernel_size = self.kernel.shape[0]
        self.half_size = self.kernel_size // 2

    def apply_filter(self, img_array, filter_function):
        if len(img_array.shape) == 3:
            height, width, _ = img_array.shape

            filtered_channels = []
            for c in range(3):
                filtered_channel = filter_function(img_array[:, :, c])
                filtered_channels.append(filtered_channel)

            filtered_image = np.zeros_like(img_array, dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    filtered_image[y, x, 0] = filtered_channels[0][y, x]
                    filtered_image[y, x, 1] = filtered_channels[1][y, x]
                    filtered_image[y, x, 2] = filtered_channels[2][y, x]

            return filtered_image
        else:
            return filter_function(img_array)

    def dilatation(self, img_array):
        # Aqui a gente expande as intensidades dos pixels
        # Expande as áreas brilhantes preenchendo lacunas e conectando objetos próximos.
        height, width = img_array.shape
        dilated = np.zeros_like(img_array)

        for y in range(self.half_size, height - self.half_size):
            for x in range(self.half_size, width - self.half_size):
                max_value = 0
                for ky in range(self.kernel_size):
                    for kx in range(self.kernel_size):
                        max_value = max(max_value, img_array[y + ky - self.half_size, x + kx - self.half_size] * self.kernel[ky, kx])
                dilated[y, x] = max_value
        return dilated

    def erosion(self, img_array):
        # Aqui a gente reduz as intensidades dos pixels brilhantes
        # Reduz as áreas brilhantes, eliminando pequenos detalhes e ruídos.
        height, width = img_array.shape
        eroded = np.zeros_like(img_array)

        for y in range(self.half_size, height - self.half_size):
            for x in range(self.half_size, width - self.half_size):
                min_value = 255
                for ky in range(self.kernel_size):
                    for kx in range(self.kernel_size):
                        min_value = min(min_value, img_array[y + ky - self.half_size, x + kx - self.half_size] * self.kernel[ky, kx])
                eroded[y, x] = min_value
        return eroded

    def morf_filter_dilatation(self, image):
        img_array = np.array(image, dtype=np.uint8)
        result = self.apply_filter(img_array, self.dilatation)
        return Image.fromarray(result)

    def morf_filter_erosion(self, image):
        img_array = np.array(image, dtype=np.uint8)
        result = self.apply_filter(img_array, self.erosion)
        return Image.fromarray(result)

    def morf_filter_open(self, image):
        eroded = self.morf_filter_erosion(image)
        opened = self.morf_filter_dilatation(eroded)
        return opened

    def morf_filter_close(self, image):
        dilated = self.morf_filter_dilatation(image)
        closed = self.morf_filter_erosion(dilated)
        return closed

    def morf_filter_contour(self, image):
        img_array = np.array(image, dtype=np.uint8)
        dilated = np.array(self.morf_filter_dilatation(image), dtype=np.uint8)
        contour = dilated - img_array
        return Image.fromarray(contour)