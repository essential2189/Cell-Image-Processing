import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from tqdm import tqdm
from collections import Counter


def contrast_stretching(img, newMin=220, newMax=255):
    img_ = img.copy()

    img_[img_ <= newMin] = newMin
    img_[img_ >= newMax] = newMax

    result = (img_ - newMin) / (newMax - newMin) * 255

    return result


def contrast_stretching2(img, newMin=15, newMax=30):
    img_ = img.copy()

    img_[img_ <= newMin] = newMin
    img_[img_ >= newMax] = newMax

    result = (img_ - newMin) / (newMax - newMin) * 255

    return result


def dam_contrast_stretching(img, newMin=250, newMax=255):
    img_ = img.copy()

    img_[img_ <= newMin] = newMin
    img_[img_ >= newMax] = newMax

    result = (img_ - newMin) / (newMax - newMin) * 255

    return result


def watershed(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    result = cv2.erode(image, kernel, iterations=1)

    result = cv2.dilate(result, kernel2, iterations=1)

    result = cv2.erode(result, kernel2, iterations=1)

    result = cv2.dilate(result, kernel, iterations=2)

    return result


def process(result, image):
    low_pass_filter = np.ones((3, 3), np.float32) / 9.0
    result = cv2.filter2D(result, -1, low_pass_filter)

    result1 = watershed(result)

    result1 = result1.astype(np.uint8)
    result1 = cv2.fastNlMeansDenoising(result1, None, 10, 7, 21)

    result1 = contrast_stretching2(result1)

    # result1 = result1.astype(np.uint8)
    # image = image.astype(np.uint8)
    # bitwise = cv2.bitwise_and(image, image, mask=result1)

    return result1


def main(path):
    files = sorted(os.listdir(path))
    files = files[:20]
    for file in tqdm(files):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        RECT = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        image = cv2.imread(path + file)
        ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        image = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)

        gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray_img = gray_img.astype(np.uint8)
        gray_img = cv2.equalizeHist(gray_img)

        result = 255 - gray_img

        plt.subplot(339)
        plt.imshow(result)

        all = contrast_stretching(result)

        dam = dam_contrast_stretching(result)

        plt.subplot(331)
        plt.imshow(all)
        plt.subplot(332)
        plt.imshow(dam)

        all_ = process(all, image)
        dam_ = process(dam, image)

        plt.subplot(333)
        plt.imshow(all_)
        plt.subplot(334)
        plt.imshow(dam_)

        dam_ = cv2.dilate(dam_, kernel, iterations=2)
        dam_ = contrast_stretching2(dam_)

        plt.subplot(335)
        plt.imshow(all_)
        plt.subplot(336)
        plt.imshow(dam_)

        nuclear = cv2.bitwise_xor(dam_, all_)
        nuclear = cv2.erode(nuclear, RECT, iterations=1)
        plt.subplot(337)
        plt.imshow(nuclear)

        nuclear = nuclear.astype(np.uint8)
        n_label, pos_label = cv2.connectedComponents(nuclear)

        label_dict = Counter([i for line in pos_label for i in line])
        label_dict[0] =0

        sort_label_list = sorted(label_dict.items(), key=lambda item:item[1], reverse=True)

        result_zero = np.zeros(pos_label.shape)
        for ins in sort_label_list:
            if ins[1] > 350:
                result_zero[pos_label == ins[0]] = 255

        plt.subplot(338)
        plt.imshow(result_zero)
        plt.tight_layout()
        plt.savefig('/home/sjwang/biotox/test/' + file + '_plt.png')
        plt.show()
        plt.clf()
        plt.cla()

        cv2.imwrite('/home/sjwang/biotox/test/' + file, result_zero)

if __name__ == '__main__':
    start = time.time()

    path = '/home/sjwang/biotox/datasets/mrxs_label/CELL1101-1/'
    # path = '/home/sjwang/biotox/datasets/image500/CELL1101-1/'
    main(path)

    end = time.time()
    print(datetime.timedelta(seconds=end-start))