import glob

import cv2

from hog import extract_carac
from transform_data import transform_images

path_bmt = 'CINTILOGRAFIAS/BMT'
path_graves = 'CINTILOGRAFIAS/GRAVES'

if __name__ == '__main__':

    transform_images(path_bmt)
    transform_images(path_graves)

    bmp_images = []
    graves_images = []

    for file in glob.glob(path_bmt + '/**/*.jpg'):
        bmp_images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

    for file in glob.glob(path_graves + '/**/*.jpg'):
        graves_images.append(cv2.imread(file, cv2.IMREAD_GRAYSCALE))

    bmp_images, graves_images = extract_carac(bmp_images, graves_images)
    print('s')
