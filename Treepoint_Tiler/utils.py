"""
Created on Fri Aug 11 12:14:47 2023
Functions used in the treepoint tiler script

Contains functions used in Treepoint Tiler main.py

@author: jlovitt
"""

import os, time, random
import cv2
import numpy as np

# Cropping Functions


def get_crop_dims(x, patch_size):
    crop = round(int(x / patch_size), 0) * patch_size
    return crop


def crop_center(
    img, cropx, cropy
):  # this function defines the central extent and crops input to match
    y, x = img.shape[:-1]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    print(startx, startx + cropx, starty, starty + cropy)
    return img[starty : starty + cropy, startx : startx + cropx]


def crop_trainset(
    img, target
):  # this function splits the image to a defined target for training and validating (eg. 80%), currently set for x clipping because input image is landscape
    y, x = img.shape[:-1]
    print(img.shape)
    startx = 0
    endx = x
    starty = int(y * (2 * target))
    endy = y
    print(startx, endx, starty, endy)
    return img[starty:endy, startx:endx]


def crop_testset(
    img, target
):  # this function splits the image to a defined target for testing (eg. 20%)
    y, x = img.shape[:-1]
    startx = 0
    endx = x
    starty = 0
    endy = int(y * target)
    print(startx, endx, starty, endy)
    return img[starty:endy, startx:endx]


def crop_valset(
    img, target
):  # this function splits the image to a defined target for validating (eg. 20%)
    y, x = img.shape[:-1]
    startx = 0
    endx = x
    starty = int(y * target)
    endy = int(y * (2 * target))
    print(startx, endx, starty, endy)
    return img[starty:endy, startx:endx]


# Dataset Preparation Functions


def prep_data(band,target, cropx, cropy):
    bc = crop_center(band,cropx, cropy)
    bc_train = crop_trainset(bc, target)
    bc_test = crop_testset(bc, target)
    bc_val = crop_valset(bc, target)
    bc_train = (bc_train * 255).astype("uint16")
    bc_test = (bc_test * 255).astype("uint16")
    bc_val = (bc_val * 255).astype("uint16")
    return bc_train, bc_test, bc_val


def split_dataset(img, target, cropx, cropy):
    img = np.einsum("ijk->kij", img)
    train_arr = []
    test_arr = []
    val_arr = []
    for i in img:
        train, test, val = prep_data(i, target, cropx, cropy)
        train_arr.append([train])
        test_arr.append([test])
        val_arr.append([val])
    train_arr = train_arr[0:3]
    # train_stk = np.vstack(train_arr)
    train_stk = np.einsum("kij->ijk", train_stk)
    test_arr = test_arr[0:3]
    # test_stk = np.vstack(test_arr)
    test_stk = np.einsum("kij->ijk", test_stk)
    val_arr = val_arr[0:3]
    # val_stk = np.vstack(val_arr)
    val_stk = np.einsum("kij->ijk", val_stk)
    return train_stk, test_stk, val_stk


def make_folders(out_path):
    if out_path == "test_environment":
        print("No folders generated in testing")
    else:
        isExist = os.path.exists(out_path + "train/")
        if not isExist:
            os.makedirs(out_path + "train/")
        isExist = os.path.exists(out_path + "train/organized/")
        if not isExist:
            os.makedirs(out_path + "train/organized/")
        isExist = os.path.exists(out_path + "test/")
        if not isExist:
            os.makedirs(out_path + "test/")
        isExist = os.path.exists(out_path + "test/organized/")
        if not isExist:
            os.makedirs(out_path + "test/organized/")
        isExist = os.path.exists(out_path + "val/")
        if not isExist:
            os.makedirs(out_path + "val/")
        isExist = os.path.exists(out_path + "val/organized/")
        if not isExist:
            os.makedirs(out_path + "val/organized/")


def check_NoData(array):
    if np.isnan(array).any():
        return -9999
    elif np.all(array == 0):
        return -9999
    else:
        return array


# Tiling Functions


def get_pts(mask):
    mask = np.array(mask)
    result = np.where(np.array(mask) == 1)
    data1 = result[1]
    data2 = result[0]
    tree_tops = zip(data1, data2)
    tree_tops = list(tree_tops)
    return tree_tops


def make_tiles(in_path, out_path, good_pts, image, annotation, patch_size, in_pad):
    patch_list = []
    conf_list = []
    in_path = os.path.abspath(in_path)
    if out_path == "test_environment":
        pass
    else:
        out_path = os.path.abspath(out_path)
    t1 = time.time()
    for i in good_pts:
        x = int(i[1])
        y = int(i[0])
        if in_pad == -1:
            pad = random.randrange(33)
            patch_size = int(128 - (pad * 2))
        else:
            pad = in_pad
        img_name = str(x) + "_" + str(y)
        suffix = str("_pad_" + str(pad))
        filename = img_name + str(suffix)
        step = int(patch_size / 2)
        x1 = x - step
        x2 = x + step
        y1 = y - step
        y2 = y + step
        img = image[x1:x2, y1:y2]
        tc_img = annotation[x1:x2, y1:y2]
        img = np.einsum("kij->jik", img)  # multi-channel input = image for tiling
        b1 = img[0]
        b2 = img[1]
        b3 = img[2]
        b1 = check_NoData(b1)
        b2 = check_NoData(b2)
        b3 = check_NoData(b3)
        if np.all(b1 != -9999) and np.all(b2 != -9999) and np.all(b3 != -9999):
            array = [b1, b2, b3]
            patch = np.dstack(array)
            padded_patch = cv2.copyMakeBorder(
                patch, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0
            )
            width, height = padded_patch.shape[:-1]
            full_tile = patch_size + (2 * pad)
            if width != full_tile:
                print("tile size invalid, width ", width)
            elif height != full_tile:
                print("tile size invalid, height ", height)
            else:  # single channel input = annotation file
                ones = np.count_nonzero(tc_img)
                patch_list.append(str(ones) + filename)
                folder = str(ones)
                out_path_join = os.path.join(str(out_path), str(folder))
                isExist = os.path.exists(out_path_join)
                if not isExist:
                    os.makedirs(out_path_join)
                img_name = str(ones) + "_" + filename + ".png"
                if out_path == "test_environment":
                    pass
                else:
                    cv2.imwrite(os.path.join(out_path_join, img_name), padded_patch)
    t2 = time.time()
    total_time = t2 - t1
    if out_path == "test_envrionment":
        return len(conf_list)
    else:
        return patch_list, total_time
