# Crop each image to patches to expand the date
import os
import shutil
import argparse
import glob
import numpy as np
from PIL import Image


def strided_crop(img, label, mask, name, height, width, m, n, l, stride=1):
    # directories = ['./data_crop/' + m + '/' + n + '/Images', './data_crop/' +
    #                m + '/' + n + '/Labels', './data_crop/' + m + '/' + n + '/Masks']
    directories = ['./data_crop/' + m + '/' + n]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    max_x = int(((img.shape[0] - height) / stride) + 1)
    # print('max_x:',max_x)
    max_y = int(((img.shape[1] - width) / stride) + 1)
    # print('max_y:',max_y)
    max_crops = max_x * max_y
    print('max_crops =', max_x, '*', max_y, '=', max_crops)
    k = 0
    for h in range(max_x):
        for w in range(max_y):
            k += 1
            crop_img_arr = img[h * stride:(h * stride) +
                               height, w * stride:(w * stride) + width]
            crop_label_arr = label[h * stride:(h * stride) +
                                   height, w * stride:(w * stride) + width]
            crop_mask_arr = mask[h * stride:(h * stride) +
                                 height, w * stride:(w * stride) + width]
            crop_img = Image.fromarray(crop_img_arr)
            crop_label = Image.fromarray(crop_label_arr)
            crop_mask = Image.fromarray(crop_mask_arr)
            # img_name = directories[0] + '/' + name + '_%03d.png' % k
            # label_name = directories[0] + '/' + \
            #     name + '_%03d_label.png' % k
            # mask_name = directories[0] + '/' + \
            #     name + '_%03d_mask.png' % k
            img_name = directories[0] + '/' + '%03d.png' % (max_crops*l+k)
            label_name = directories[0] + '/' + \
                '%03d_label.png' % (max_crops*l+k)
            mask_name = directories[0] + '/' + \
                '%03d_mask.png' % (max_crops*l+k)
            crop_img.save(img_name)
            crop_label.save(label_name)
            # crop_mask.save(mask_name)


if __name__ == '__main__':

    dataset = ['CHASE', 'DRIVE']
    train_test = ['train', 'test']

    # Copy 1st_manual from 'data' folder to 'data_process' folder
    source_path = ['./data/'+dataset[i]+'/'+train_test[j]+'/1st_manual'
                   for i in range(len(dataset)) for j in range(len(train_test))]
    target_path = ['./data_process/'+dataset[i]+'/'+train_test[j]+'/1st_manual'
                   for i in range(len(dataset)) for j in range(len(train_test))]
    print(source_path, target_path)
    for i in range(4):
        if not os.path.exists(target_path[i]):
            os.makedirs(target_path[i])
        if os.path.exists(source_path[i]):
            shutil.rmtree(target_path[i])
        shutil.copytree(source_path[i], target_path[i])
    print('Copy dir finished!')

    # Set some parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--side_length', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    args = parser.parse_args()

    # Crop from images
    for m in dataset:
        for n in train_test:
            img_dir = './data_process/' + m + '/' + n + '/preprocessing/'
            img_name = os.listdir(img_dir)
            images = [img_dir+img_name[i] for i in range(len(img_name))]
            l = 0
            for i, j in enumerate(images):
                image_name = j.split('/')[-1].split('.')[0][:-6]
                # print(image_name) # Image_01L, ..., 20_test
                im = Image.open(images[i])
                img_arr = np.asarray(im)
                if m == 'CHASE':
                    label_name = './data_process/' + m + '/' + n + \
                        '/1st_manual/' + image_name + '_1stHO.png'
                elif m == 'DRIVE':
                    label_name = './data_process/' + m + '/' + n + \
                        '/1st_manual/' + image_name[:2] + '_manual1.gif'
                label = Image.open(label_name)
                label_arr = np.asarray(label)
                mask_name = './data_process/' + m + '/' + \
                    n + '/mask/' + image_name + '_mask.jpg'
                mask = Image.open(mask_name)
                mask_arr = np.asarray(mask)
                strided_crop(img_arr, label_arr, mask_arr, image_name,
                             args.side_length, args.side_length, m, n, l, args.stride)
                l += 1
