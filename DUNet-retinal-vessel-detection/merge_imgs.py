import os
import numpy as np
from PIL import Image

# No1toChange.merge predict
# source_path = ['./result/CHASE/test/', './result/DRIVE/test/']
# merge_path = ['./result/CHASE/merge/', './result/DRIVE/merge/']
# No1toChange.merge mask. Why don not we use raw mask? Because the size of merged croped mask & raw mask are not equail.
source_path = ['./data_crop/CHASE/test/', './data_crop/DRIVE/test/']
merge_path = ['./result/CHASE/merge_label/', './result/DRIVE/merge_label/']
for i in range(len(merge_path)):
    if not os.path.exists(merge_path[i]):
        os.makedirs(merge_path[i])


def merge(path_in, path_out, row, col, idx, side_length, stride, size):
    merged_img_arr = np.zeros(size)
    l = 0
    for i in range(row):
        for j in range(col):
            l += 1
            # print(path_in+'%03d_predict.png' %
            #       (idx*row*col+l))
            # No2toChange.predict patches path
            # img_ij = Image.open(path_in+'%03d_predict.png' %
            #                     (idx*row*col+l))
            # No2toChange.label patches path
            img_ij = Image.open(path_in+'%03d_label.png' %
                                (idx*row*col+l))
            img_ij = np.asarray(img_ij)
            # print(img_ij.shape)
            print(str(idx+1)+'-'+str(l))
            if idx == 0 and l == 1:
                print(img_ij)
            print([i*stride, i*stride+side_length,
                   j*stride, j*stride+side_length])
            # No3toChange. CHASE_prediction*1 or DRIVE_prediction*1
            # merged_img_arr[int(i*stride):int(i*stride+side_length),
            #                int(j*stride):int(j*stride+side_length)] = img_ij
            # No3toChange. CHASE_label*255 or DRIVE_label*1
            # merged_img_arr[int(i*stride):int(i*stride+side_length),
            #                int(j*stride):int(j*stride+side_length)] = img_ij*255
            merged_img_arr[int(i*stride):int(i*stride+side_length),
                           int(j*stride):int(j*stride+side_length)] = img_ij
    merged_img = Image.fromarray(merged_img_arr).convert('L')
    merged_img.save(path_out+'%03d.png' % (idx+1))


if __name__ == '__main__':
    
    # No4toChange. CHASE 0
    # dataset = 0
    # No4toChange. DRIVE 1
    dataset = 1
    side_length = 256
    stride = 128
    path_in = source_path[dataset]
    path_out = merge_path[dataset]
    # No5toChange.predict
    # files_num = len(os.listdir(source_path[dataset]))
    # No5toChange.label
    files_num = len(os.listdir(source_path[dataset]))//2
    if dataset == 0:
        row, col = 6, 6
        # size = (999, 960)
        size = (6*stride+side_length-stride, 6*stride+side_length-stride)
    elif dataset == 1:
        row, col = 3, 3
        # size = (565, 584)
        size = (3*stride+side_length-stride, 3*stride+side_length-stride)
    idx_all = files_num/row/col

    for idx in range(int(idx_all)):
        merge(path_in, path_out, int(row), int(col),
              int(idx), side_length, stride, size)
