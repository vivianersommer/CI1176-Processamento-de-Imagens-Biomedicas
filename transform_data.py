import glob

import cv2
import pydicom as dicom


def transform_images(path):
    for file in glob.glob(path + '/**/*.dcm'):
        ds = dicom.dcmread(file)
        dcm_to_jpg(ds, file)


def dcm_to_jpg(ds, file):
    pixel_array_numpy = ds.pixel_array
    file = file.replace('.dcm', '.jpg')
    cv2.imwrite(file, pixel_array_numpy)
