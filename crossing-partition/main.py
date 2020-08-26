import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy
from skimage import io
import math
import copy
import time
from utils import *

IMG_SIZE = 256
IMAGE_PATH = './img'
OVERLAP_PATH = './crossing'
OTHER_PATH = './chromosome'
OUTPUT_PATH = './output'
FILE_NUM = 2

if __name__ == "__main__":
    for i in range(1, FILE_NUM + 1):
        image = cv2.imread(os.path.join(IMAGE_PATH, str(i)+'.png'))
        overlapped = cv2.imread(os.path.join(OVERLAP_PATH, str(i)+'.png'), 0)
        foreground = cv2.imread(os.path.join(OTHER_PATH, str(i)+'.png'), 0)
        non_overlapped = np.zeros(overlapped.shape)
        # kernel = np.ones((1,1),np.uint8)
        # overlapped = cv2.dilate(overlapped,kernel,iterations=1)
        overlapped[overlapped == 255] = 1
        non_overlapped[(overlapped == 0) & (foreground == 255)] = 1   # calculate non-overlapping mask 

        output = crossing_reconstruct(image, overlapped, non_overlapped)

        output_path = os.path.join(OUTPUT_PATH, str(i))
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        cv2.imwrite(os.path.join(output_path, "crossing_" + str(i)+".png"), image)

        try:
            gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            num = 0
            for mask in output:
                img = copy.deepcopy(gray)
                new_mask = np.zeros((IMG_SIZE, IMG_SIZE))
                for i in range(0, IMG_SIZE):
                    for j in range(0, IMG_SIZE):
                        if(mask[i,j] == 1):
                            for x in range(-1,2):
                                for y in range(-1,2):
                                    try:
                                        new_mask[i+x,j+y] = 1
                                    except:
                                        continue

                img[(new_mask == 0)] = 255
                cv2.imwrite(os.path.join(output_path, str(num)+".png"), img)
                num += 1
        except:
            raise "error(fail to partition)"




