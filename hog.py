import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog


def extract_carac(images):

    images_hog = []
    for image in images:

        fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), visualize=True)

        images_hog.append(hog_image)

        # Descomentar para ver imagem com HOG
        plt.axis("off")
        plt.imshow(hog_image, cmap="gray")
        plt.show()
    return images_hog
