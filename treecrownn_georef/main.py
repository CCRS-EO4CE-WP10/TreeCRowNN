# import numpy as np
import rasterio
from osgeo import gdal
import click

driver = gdal.GetDriverByName("GTiff")


@click.command()
@click.argument("geo_path", type=click.Path(exists=True))
@click.argument("data_path", type=click.Path(exists=True))
@click.argument("out_path", type=click.Path(exists=True))
@click.option(
    "--tile_size",
    "--ts",
    default=128,
    type=int,
    help="value of data spatial resolution (int). default 128",
)
@click.option(
    "--is_FSD",
    "--fsd",
    default=True,
    type=bool,
    help="True = data is FSD map, False = data is activation map",
)
def get_window_crop(
    img, tile_size
):  # this function crops the image with metadata to match inference extents, as per inference cropping process
    # from rasterio docs: Window(col_off, row_off, width, height)
    x = img.width
    y = img.height
    cropx = round(int(x / tile_size), 0) * tile_size
    cropy = round(int(y / tile_size), 0) * tile_size
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    print(startx, startx + cropx, starty, starty + cropy)
    window = Window(startx, starty, cropx, cropy)
    return window


def main(geo_path, data_path, out_path, tile_size, is_FSD):
    raster_with_ref = rasterio.open(geo_path)
    input_crs = raster_with_ref.crs
    input_gt = raster_with_ref.transform
    print(input_crs)
    print(
        "image width: ",
        raster_with_ref.width,
        "  image height: ",
        raster_with_ref.height,
    )
    print(
        "Upper Left Image Origin= X:",
        input_gt[2],
        ", Y:",
        input_gt[5],
        "Pixel size: ",
        input_gt[0],
        "m,",
        input_gt[4],
        "m",
    )
    ras = get_window_crop(raster_with_ref, tile_size)
    newxo = input_gt[2] + (((raster_with_ref.width - ras.width) / 2) * input_gt[0])
    newyo = input_gt[5] + (((raster_with_ref.height - ras.height) / 2) * input_gt[4])

    if is_FSD == True:
        ground_resolution = tile_size / 10
    else:
        ground_resolution = input_gt[0]

    rescale = rasterio.transform.from_origin(
        newxo, newyo, ground_resolution, ground_resolution
    )
    raster_with_data = rasterio.open(data_path)
    raster_array_with_data = raster_with_data.read()
    dtype = raster_array_with_data.dtype

    if is_FSD == True:
        with raster_with_ref as src:
            kwargs = raster_with_ref.meta.copy()
            print(kwargs)

            kwargs.update(
                {
                    "height": raster_with_data.height,
                    "width": raster_with_data.width,
                    "dtype": dtype,
                    "count": 1,
                    "transform": rescale,
                }
            )
            print(kwargs)

            with rasterio.open(
                out_path, "w", **kwargs
            ) as dst:  
                dst.write(raster_with_data.read())

    else:
        with raster_with_ref as src:
            kwargs = raster_with_ref.meta.copy()
            print(kwargs)
            kwargs.update(
                {
                    "height": raster_with_data.height,
                    "width": raster_with_data.width,
                    "dtype": dtype,
                    "count": 1,
                    "transform": src.transform,
                }
            )
            print(kwargs)
            with rasterio.open(
                out_path, "w", **kwargs
            ) as dst:  
                dst.write(raster_with_data.read())


# pylint: disable = no-value-for-parameter
if __name__ == "__main__":
    main()
