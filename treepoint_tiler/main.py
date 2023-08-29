#!/usr/bin/env python

import tifffile as tiff
import numpy as np
import click
import utils as ut


@click.command()
@click.argument("img_path", type=click.Path(exists=True))
@click.argument("mask_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(exists=True))
@click.option(
    "--NoData",
    "--nd",
    default=-9999,
    type=float,
    help="no data value of input image (float). default -9999",
)
@click.option(
    "--target",
    "--tg",
    default=0.2,
    type=float,
    help="desired dataset split (float). default 0.2 gives 60/20/20% train/test/val",
)
@click.option(
    "--patch_size",
    "--ps",
    default=128,
    type=int,
    help="enter -1 to generate random tile sizes",
)
@click.option(
    "--padding",
    "--pad",
    default=0,
    type=int,
    help="enter -1 to generate tiles with random padding",
)
def main(img_path, mask_path, out_path, NoData, target, patch_size, padding):
    """This tool will generate 3 channel image tiles from an input image using annotations
    as centerpoint.
    It requires an image for tiling (img_path),
    an annotation file (mask_path) and output folder to write (out_path).

    """
    if out_path == "test_environment":
        large_image_stack = img_path
        large_mask_file = mask_path

    else:
        large_image_stack = tiff.imread(img_path)
        large_mask_file = tiff.imread(mask_path).astype(np.uint16)
    print(
        "MULTI-CHANNEL IMAGE SHAPE : ",
        large_image_stack.shape,
        " MIN ",
        np.min(large_image_stack),
        " MAX ",
        np.max(large_image_stack),
    )
    print(
        "ANNOTATION FILE SHAPE : ",
        large_mask_file.shape,
        " MIN ",
        np.min(large_mask_file),
        " MAX ",
        np.max(large_mask_file),
    )
    print("PREPARING DATASETS")
    ut.make_folders(out_path)
    y, x = large_mask_file.shape
    cropx = ut.get_crop_dims(x, patch_size)
    cropy = ut.get_crop_dims(y, patch_size)
    print("crop_x: ", cropx, "crop_y: ", cropy)
    train_stk, test_stk, val_stk = ut.split_dataset(
        large_image_stack, target, cropx, cropy
    )  # RGB image prep
    large_mask_file[large_mask_file == NoData] = 0
    tc_c = ut.crop_center(large_mask_file, cropx, cropy)
    tc_train = ut.crop_trainset(tc_c, target)
    tc_test = ut.crop_testset(tc_c, target)
    tc_val = ut.crop_valset(tc_c, target)
    tc_c = None
    del tc_c

    print(
        "TRAINING SET   (y,x)   ==>   RGB:",
        train_stk.shape,
        ", Tree Count:",
        tc_train.shape,
    )
    print(
        "TESTING SET    (y,x)   ==>   RGB:",
        test_stk.shape,
        ", Tree Count:",
        tc_test.shape,
    )
    print(
        "VALIDATION SET (y,x)   ==>   RGB:",
        val_stk.shape,
        ", Tree Count:",
        tc_val.shape,
    )
    print("PROCESSING DATASETS...")
    # make train tileset
    out1 = out_path + "train/"
    out2 = out1 + "organized/"
    tree_pts = ut.get_pts(tc_train)
    print("TOTAL ANNOTATIONS FOUND : ", len(tree_pts))
    patch_list, t3 = ut.make_tiles(
        out1, out2, tree_pts, train_stk, tc_train, patch_size, padding
    )
    train_tiles = len(patch_list)
    print(
        train_tiles,
        " TRAINING TILES GENERATED AND ORGANIZED IN : ",
        round(t3, 5),
        " SECONDS",
    )
    print("TOTAL PROCESSING TIME PER TILE : ", round(t3 / train_tiles, 5), " SECONDS")
    # make test tileset
    out1 = out_path + "test/"
    out2 = out1 + "organized/"
    tree_pts = ut.get_pts(tc_test)
    print("TOTAL ANNOTATIONS FOUND : ", len(tree_pts))
    patch_list, t5 = ut.make_tiles(
        out1, out2, tree_pts, test_stk, tc_test, patch_size, padding
    )
    test_tiles = len(patch_list)
    print(
        test_tiles,
        " TESTING TILES GENERATED AND ORGANIZED IN : ",
        round(t5, 5),
        " SECONDS",
    )
    print("TOTAL PROCESSING TIME PER TILE : ", round(t5 / test_tiles, 5), " SECONDS")
    # make val tileset
    out1 = out_path + "val/"
    out2 = out1 + "organized/"
    tree_pts = ut.get_pts(tc_val)
    print("TOTAL ANNOTATIONS FOUND : ", len(tree_pts))
    patch_list, t7 = ut.make_tiles(
        out1, out2, tree_pts, val_stk, tc_val, patch_size, padding
    )
    val_tiles = len(patch_list)
    print(
        val_tiles,
        " VALIDATION TILES GENERATED AND ORGANIZED IN : ",
        round(t7, 5),
        " SECONDS",
    )
    print("TOTAL PROCESSING TIME PER TILE : ", round(t7 / val_tiles, 5), " SECONDS")
    print(
        "TOTAL TRAIN TILES : ",
        train_tiles,
        " TOTAL TEST TILES : ",
        test_tiles,
        " TOTAL VALIDATION TILES : ",
        val_tiles,
    )
    print("END PROGRAM : ", (train_tiles + test_tiles + val_tiles), " TILES GENERATED")


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    main()
