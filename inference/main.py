#!/usr/bin/env python

import utils as ut
import os
import numpy as np
import tifffile as tiff
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf

print(tf.__version__)
print(f"Tensorflow ver. {tf.__version__}")
physical_device = tf.config.experimental.list_physical_devices("CPU")
print(f"Device found : {physical_device}")
import click


@click.command()
@click.argument("img_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(exists=True))
@click.option(
    "--site",
    default="in1",
    type=str,
    help="name to be given to model output. default in1",
)
@click.option(
    "--model",
    "--mod",
    default="v1",
    type=str,
    help="enter either v1 or v2 to choose TreeCRowNN model. default v1",
)
@click.option(
    "--NoData",
    "--nd",
    default=-9999,
    type=int,
    help="value of NoData in img file",
)
@click.option(
    "--tile_size",
    "--ts",
    default=128,
    type=int,
    help="default 128",
)
@click.option(
    "--batch_size",
    "--bs",
    default=100,
    type=int,
    help="default 100",
)
@click.option(
    "--minval",
    "--min",
    default=0,
    type=int,
    help="default 0",
)
@click.option(
    "--maxval",
    "--max",
    default=255,
    type=int,
    help="default 255",
)
@click.option(
    "--generate_heatmap",
    "--gh",
    default=False,
    type=bool,
    help="True : generate activation map in addition to FSD map (increase processing time ~4x). default False",
)
@click.option(
    "--normalize_heatmap",
    "--nh",
    default="none",
    type=str,
    help="Enter either : local, global, none. default none",
)
def main(
    img_path,
    out_path,
    site,
    model,
    NoData,
    tile_size,
    batch_size,
    minval,
    maxval,
    generate_heatmap,
    normalize_heatmap,
):
    padding = int((128 - tile_size) / 2)
    print(f"padding : {padding}")
    img = tiff.imread(img_path)
    print(img.min(), img.max())
    img = np.einsum("ijk->kij", img)
    img[img == NoData] = np.nan
    b1 = img[0]
    b2 = img[1]
    b3 = img[2]
    b1 = ut.crop_image(b1, tile_size)
    b2 = ut.crop_image(b2, tile_size)
    b3 = ut.crop_image(b3, tile_size)
    array = [b3, b2, b1]
    img = np.dstack(array)
    b1 = b2 = b3 = array = None
    del (b1, b2, b3, array)
    print(img.min(), img.max())
    y, x, chan = img.shape
    big_x = round(int(x / tile_size), 0)#cropx = big_x * tile_size
    big_y = round(int(y / tile_size), 0)#cropy = big_y * tile_size
    blank_img = np.zeros([big_y, big_x], dtype=int)
    blank_img2 = np.zeros([y, x], dtype=np.float32)
    print(blank_img.shape, blank_img2.shape)

    #model selection process pending
    if model == v1:
        model = tc_model1(lr, spe, u, u2, u3)
        model.load_weights(path_to_weights1)
    elif model == v2:
        model = tc_model2(lr, spe, u, u2, u3)
        model.load_weights(path_to_weights2)

    T2 = ut.create_image(
        img,
        blank_img,
        blank_img2,
        tile_size,
        model,
        chan,
        padding,
        minval,
        maxval,
        batch_size,
        "batch_normalization_23",
        generate_heatmap,
        normalize_heatmap,
    )
    print((big_y * big_x), " BLOCKS PROCESSED IN ", round(T2, 3), " SECONDS")
    print("TOTAL TIME PER BLOCK : ", (T2 / (big_y * big_x)), "sec")
    total_trees = str(np.ndarray.sum(blank_img))
    m = tile_size / 10
    area = round(((big_y * big_x) * (m * m)), 2)
    print(
        "TOTAL TREES PREDICTED WITH TILE SIZE " + str(tile_size) + " : " + total_trees
    )
    print("(COL, ROW)", blank_img.shape)
    print("TOTAL AREA PROCESSED : ", area, "m2 --> ", round(area / 10000, 2), "ha")
    print("TOTAL PROCESSING TIME : ", round(T2 / 60, 1), "MINUTES")
    print("PREDICTED RANGE : ", blank_img.min(), " to ", blank_img.max(), " TREES")
    # write FSD map
    tiff.imwrite(
        str(os.path.join(out_path + site + "_totaltrees_" + total_trees + ".tif")),
        blank_img,
    )
    # write activation map if requested
    if generate_heatmap == True:
        if normalize_heatmap == "local":
            tiff.imwrite(
                str(os.path.join(out_path + site + "_heatmap_localNorm" + ".tif")),
                blank_img2,
            )
        elif normalize_heatmap == "global":
            n_heatmap = normalize_activations(blank_img2)
            tiff.imwrite(
                str(os.path.join(out_path + site + "_heatmap_globalNorm" + ".tif")),
                n_heatmap,
            )
        elif normalize_heatmap == "none":
            tiff.imwrite(
                str(os.path.join(out_path + site + "heatmap_noNorm" + ".tif")),
                blank_img2,
            )


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    main()
