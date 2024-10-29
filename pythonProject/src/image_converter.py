from PIL import Image
import numpy as np
import os

np.set_printoptions(threshold=np.inf)


def load_image(image_path):
    img = Image.open(image_path)
    img_array = np.array(img)
    return img_array


def sum_images(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("As imagens devem ter o mesmo tamanho")

    sum_img = np.zeros_like(image1, dtype=np.int32)

    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
                pixel_sum = image1[i, j] + image2[i, j]
                sum_img[i, j] = pixel_sum

    sum_img = sum_img.astype(np.uint8)

    return sum_img


def save_image(image_array, path):
    img = Image.fromarray(image_array)
    img.save(path)
    print(f"Imagem salva em: {path}")


image_path1 = os.path.join(os.path.dirname(__file__), '../src/images/Imagem1.tif')
image_path2 = os.path.join(os.path.dirname(__file__), '../src/images/Imagem1.tif')

img1_array = load_image(image_path1)
img2_array = load_image(image_path2)

summed_img = sum_images(img1_array, img2_array)

output_image_path = os.path.join(os.path.dirname(__file__), '../src/summed_image.png')
save_image(summed_img, output_image_path)
