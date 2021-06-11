import os
import numpy as np
from PIL import Image


def Dice(y_true, y_pred):
    # input is array
    if np.max(y_true) > 1:
        y_true = y_true / 255.
    if np.max(y_pred) > 1:
        y_pred = y_pred / 255.
    return 2 * np.sum(y_true * y_pred) / (np.sum(y_pred) + np.sum(y_true))


def Accuracy(y_true, y_pred):
    # input is array
    if np.max(y_true) > 1:
        y_true = y_true / 255.
    if np.max(y_pred) > 1:
        y_pred = y_pred / 255.
    acc_sum = 0
    sum = 0
    for i in range(len(y_true)):
        for j in range(len(y_true[i])):
            sum += 1
            if y_true[i][j] == y_pred[i][j]:
                acc_sum += 1
    accuracy = acc_sum/sum
    return accuracy


if __name__ == '__main__':
    CHASE_predict_path = './result/CHASE/merge/'
    CHASE_label_path = './result/CHASE/merge_label/'
    DRIVE_predict_path = './result/DRIVE/merge/'
    DRIVE_label_path = './result/DRIVE/merge_label/'

    Dice_CHASE_sum = 0
    Accuracy_CHASE_sum = 0
    num_CHASE = len(os.listdir(CHASE_predict_path))
    for i in range(num_CHASE):
        x = np.asarray(Image.open(CHASE_predict_path+'%03d.png' % (i+1)))
        y = np.asarray(Image.open(CHASE_label_path+'%03d.png' % (i+1)))
        Dice_i = Dice(x, y)
        Accuracy_i = Accuracy(x, y)
        print('CHASE No.{}, Dice:{:.2f}%, Accuracy:{:.2f}%'.format(
            i+1, Dice_i*100, Accuracy_i*100))
        Dice_CHASE_sum += Dice_i
        Accuracy_CHASE_sum += Accuracy_i
    avg_dice = Dice_CHASE_sum*100/num_CHASE
    avg_accuracy = Accuracy_CHASE_sum*100/num_CHASE
    print('CHASE avarage Dice:{:.2f}%, avarage Accuracy:{:.2f}%'.format(
        avg_dice, avg_accuracy))

    Dice_DRIVE_sum = 0
    Accuracy_DRIVE_sum = 0
    num_DRIVE = len(os.listdir(DRIVE_predict_path))
    for i in range(num_DRIVE):
        x = np.asarray(Image.open(DRIVE_predict_path+'%03d.png' % (i+1)))
        y = np.asarray(Image.open(DRIVE_label_path+'%03d.png' % (i+1)))
        Dice_i = Dice(x, y)
        Accuracy_i = Accuracy(x, y)
        print('DRIVE No.{}, Dice:{:.2f}%, Accuracy:{:.2f}%'.format(
            i+1, Dice_i*100, Accuracy_i*100))
        Dice_DRIVE_sum += Dice_i
        Accuracy_DRIVE_sum += Accuracy_i
    avg_dice = Dice_DRIVE_sum*100/num_DRIVE
    avg_accuracy = Accuracy_DRIVE_sum*100/num_DRIVE
    print('DRIVE avarage Dice:{:.2f}%, avarage Accuracy:{:.2f}%'.format(
        avg_dice, avg_accuracy))
