from tkinter import Image

import numpy as np
class LogicalOperations:

    @staticmethod
    def and_operation(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("As imagens devem ter o mesmo tamanho")

        result_img = np.zeros_like(img1, dtype=np.uint8)

        if img1.ndim == 3:
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    for k in range(img1.shape[2]):
                        result_img[i, j, k] = img1[i, j, k] & img2[i, j, k]
        else:
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    result_img[i, j] = img1[i, j] & img2[i, j]

        return result_img

    @staticmethod
    def or_operation(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("As imagens devem ter o mesmo tamanho")

        result_img = np.zeros_like(img1, dtype=np.uint8)

        if img1.ndim == 3:
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    for k in range(img1.shape[2]):
                        result_img[i, j, k] = img1[i, j, k] | img2[i, j, k]
        else:
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    result_img[i, j] = img1[i, j] | img2[i, j]

        return result_img

    @staticmethod
    def xor_operation(img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("As imagens devem ter o mesmo tamanho")

        result_img = np.zeros_like(img1, dtype=np.uint8)

        if img1.ndim == 3:
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    for k in range(img1.shape[2]):
                        result_img[i, j, k] = img1[i, j, k] ^ img2[i, j, k]
        else:
            for i in range(img1.shape[0]):
                for j in range(img1.shape[1]):
                    result_img[i, j] = img1[i, j] ^ img2[i, j]

        return result_img

    @staticmethod
    def not_operation(img):
        result_img = np.zeros_like(img, dtype=np.uint8)
        if img.ndim == 3:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    for k in range(img.shape[2]):
                        result_img[i, j, k] = ~img[i, j, k]
        else:
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    result_img[i, j] = ~img[i, j]


        return result_img
