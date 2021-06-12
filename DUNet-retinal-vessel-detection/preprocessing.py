# 4 steps to enhance image
import cv2
import os
import numpy as np


# Step1.Convert RGB to gray
def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :]*0.299 + \
        rgb[:, 1, :, :]*0.587 + rgb[:, 2, :, :]*0.114
    bn_imgs = np.reshape(
        bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


# Step2.Normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
            np.max(imgs_normalized[i])-np.min(imgs_normalized[i])))*255
    return imgs_normalized


# Step3.CLAHE (Contrast Limited Adaptive Histogram Equalization)
# In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied.
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)
    assert (imgs.shape[1] == 1)
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(
            np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# Step4.Adjust gamma
def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)
    assert (imgs.shape[1] == 1)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) *
                      255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


# Combine process: input: imgs[image_number, channel, height, width]
def combine_process(imgs):
    assert(len(imgs.shape) == 4)
    assert(imgs.shape[1] == 3)
    # global gray_imgs, normalized_imgs, clahe_imgs, gamma_imgs, finally_imgs
    gray_imgs = rgb2gray(imgs)
    normalized_imgs = dataset_normalized(gray_imgs)
    clahe_imgs = clahe_equalized(normalized_imgs)
    gamma_imgs = adjust_gamma(clahe_imgs, 1.2)
    finally_imgs = gamma_imgs/255.  # reduce to 0-1 range
    return gray_imgs, normalized_imgs, clahe_imgs, gamma_imgs, finally_imgs


# Main process
def main_process():

    # Step0.Set original & preprocessed image path
    dataset = ['CHASE', 'DRIVE']
    train_test = ['train', 'test']
    img_folder_path = ['./data/'+dataset[i]+'/' + train_test[j] +
                       '/images/' for i in range(2) for j in range(2)]
    img_name = [os.listdir(img_folder_path[i]) for i in range(4)]
    img_path = [[img_folder_path[i]+img_name[i][j]
                 for j in range(len(img_name[i]))] for i in range(4)]
    # print(img_path) # [[20img_path],[8],[20],[20]]
    preprocessing_folder_path = ['./data_process/'+dataset[i]+'/' + train_test[j] +
                                 '/preprocessing/' for i in range(2) for j in range(2)]
    for val in preprocessing_folder_path:
        if not os.path.exists(val):
            os.makedirs(val)

    def new_path(name):
        return [[preprocessing_folder_path[i]+os.path.splitext(img_name[i][j])[0]+'_'+name+'.jpg'
                 for j in range(len(img_name[i]))] for i in range(4)]
    gray_path = new_path('gray')
    normalized_path = new_path('normalized')
    clahe_path = new_path('clahe')
    gamma_path = new_path('gamma')
    finally_path = new_path('finally')
    gray_imgs, normalized_imgs, clahe_imgs, gamma_imgs, finally_imgs = [
        [[] for x in range(4)] for y in range(5)]

    # 1).Read images using cv2; bgr -> rgb; (0, 1, 2, 3) -> (0, 3, 1, 2)(img_number, channel, height, width)
    imgs = [np.array([cv2.imread(img_path[i][j])
                      for j in range(len(img_path[i]))]) for i in range(4)]
    for i in range(len(imgs)):
        for j in range(len(imgs[i])):
            B, G, R = cv2.split(imgs[i][j])
            img_merge = cv2.merge((R, G, B))
            imgs[i][j] = img_merge

    def change_dim(which, order):
        return [np.transpose(which[i], order) for i in range(4)]
    imgs = change_dim(imgs, (0, 3, 1, 2))
    # print(len(imgs))
    # for i in range(4):
    #     print(imgs[i].shape)
    # 4 (20, 3, 960, 999) (8, 3, 960, 999) (20, 3, 584, 565) (20, 3, 584, 565)

    # 2).Step1~4: rgb2gray, dataset_normalized, clahe_equalized, adjust_gamma
    # global gray_imgs, normalized_imgs, clahe_imgs, gamma_imgs, finally_imgs
    for i in range(4):
        gray_imgs[i], normalized_imgs[i], clahe_imgs[i], gamma_imgs[i], finally_imgs[i] = combine_process(
            imgs[i])

    # 3).Step5~8: save img_gray, img_normalized, img_clahe, img_clahe, img_gamma, gamma00.txt, finally00.txt
    gray_imgs = change_dim(gray_imgs, (0, 2, 3, 1))
    normalized_imgs = change_dim(normalized_imgs, (0, 2, 3, 1))
    clahe_imgs = change_dim(clahe_imgs, (0, 2, 3, 1))
    gamma_imgs = change_dim(gamma_imgs, (0, 2, 3, 1))
    finally_imgs = change_dim(finally_imgs, (0, 2, 3, 1))
    for i in range(len(gray_imgs)):
        for j in range(len(gray_imgs[i])):
            # print(gray_imgs[i][j].shape) # (960, 999, 1)*2*2, (584, 565, 1)*2*2
            # cv2.imwrite(gray_path[i][j], gray_imgs[i][j])
            # cv2.imwrite(normalized_path[i][j], normalized_imgs[i][j])
            # cv2.imwrite(clahe_path[i][j], clahe_imgs[i][j])
            cv2.imwrite(gamma_path[i][j], gamma_imgs[i][j])
            # cv2.imwrite(finally_path[i][j], finally_imgs[i][j]) # The images(0~1) will be black to see, so we haven't saven them.
    # np.savetxt('gamma00.txt', gamma_imgs[0][0]
    #            [:, :, 0], fmt="%.3f", delimiter=',')
    # np.savetxt('finally00.txt', finally_imgs[0][0]
    #            [:, :, 0], fmt="%.3f", delimiter=',')
    return gamma_imgs, finally_imgs


if __name__ == '__main__':

    gamma_imgs, finally_imgs = main_process()
    print(type(gamma_imgs), len(gamma_imgs),
          gamma_imgs[0].shape, gamma_imgs[0][0].shape)
