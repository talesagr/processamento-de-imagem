import numpy as np

class EdgeFilter:
    def sobel(self, img_array):
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=np.float32)

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=np.float32)

        return self.calculate_rgb(img_array, sobel_x, sobel_y)

    def laplacian(self, img_array):
        laplacian_kernel = np.array([[0, -1, 0],
                                     [-1, 4, -1],
                                     [0, -1, 0]], dtype=np.float32)

        return self.calculate_rgb(img_array, laplacian_kernel, None)

    def prewitt(self, img_array):
        prewitt_x = np.array([[-1, 0, 1],
                              [-1, 0, 1],
                              [-1, 0, 1]], dtype=np.float32)

        prewitt_y = np.array([[-1, -1, -1],
                              [0, 0, 0],
                              [1, 1, 1]], dtype=np.float32)

        return self.calculate_rgb(img_array, prewitt_x, prewitt_y)

    def calculate_rgb(self, img_array, kernel_x, kernel_y=None):
        height, width, channels = img_array.shape
        output = np.zeros_like(img_array, dtype=np.float32)

        for c in range(channels):
            channel = img_array[:, :, c]
            if kernel_y is not None:
                output[:, :, c] = self.calculate(channel, kernel_x, kernel_y)
            else:
                output[:, :, c] = self.calculate_single_kernel(channel, kernel_x)

        return np.clip(output, 0, 255).astype(np.uint8)

    def calculate(self, channel, kernel_x, kernel_y):
        height, width = channel.shape
        output = np.zeros_like(channel, dtype=np.float32)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                gx = 0
                gy = 0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        pixel = channel[y + ky, x + kx]
                        gx += pixel * kernel_x[ky + 1, kx + 1]
                        gy += pixel * kernel_y[ky + 1, kx + 1]
                output[y, x] = np.sqrt(gx ** 2 + gy ** 2)

        return output

    def calculate_single_kernel(self, channel, kernel):
        height, width = channel.shape
        output = np.zeros_like(channel, dtype=np.float32)

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                value = 0
                for ky in range(-1, 2):
                    for kx in range(-1, 2):
                        pixel = channel[y + ky, x + kx]
                        value += pixel * kernel[ky + 1, kx + 1]
                output[y, x] = abs(value)

        return output
