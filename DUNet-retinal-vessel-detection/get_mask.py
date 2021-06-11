# Get masks of each image
import os
import numpy as np
from PIL import Image


if __name__ == '__main__':

    dataset = ['CHASE', 'DRIVE']
    train_test = ['train', 'test']
    for m in dataset:
        for n in train_test:
            img_dir = './data/' + m + '/' + n + '/images/'
            img_name = os.listdir(img_dir)
            images = [img_dir+img_name[i] for i in range(len(img_name))]
            # print(images)
            directory = './data_process/' + m + '/' + n + '/mask/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            for i in images:
                image_name = i.split('/')[-1].split('.')[0]
                # print(image_name)
                im = Image.open(i)
                im_gray = im.convert('L')
                np_im = np.array(im_gray)
                np_mask = np.zeros((np_im.shape[0], np_im.shape[1]))
                if m == 'CHASE':
                    np_mask[np_im[:, :] > 5] = 255
                elif m == 'DRIVE':
                    np_mask[np_im[:, :] > 20] = 255
                mask = Image.fromarray(np_mask)
                mask = mask.convert('L')
                mask_name = directory + image_name + '_mask.jpg'
                mask.save(mask_name)
