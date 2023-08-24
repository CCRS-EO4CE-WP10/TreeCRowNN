# TreeCRowNN Project
Welcome to the Tree Convolutional Row Neural Network (Tree-CRowNN) Project! </br>
</br>This project supports estimations of forest stand density (FSD) from high-resolution RGB aerial imagery.
Imagery used in the initial model development is plane-based 10cm spatial resolution, collected over a mountainous region of interior BC in October 2019.
### The project contains three modules to support: 
- Tile Generation
- Model Inference
- Georeferencing Output
## [License](https://github.com/JulieLovitt/TreeCRowNN/blob/main/LICENSE)

# [Treepoint Tiler Module](https://github.com/JulieLovitt/TreeCRowNN/tree/main/Treepoint_Tiler)
This module will generate tiles from an image that can then be used in model development or transfer learning. 
<br/>It accepts two input: an image you wish to extract tiles from and a complementary binary raster (annotation file).
<br/> It will: 
- Divide the image to train,test,val datasets based on target %.
- Extract tiles from image and add padding (if desired).
- Extract tiles from annotation file and compute sum of 1s for each to determine tree counts.
- Move image tiles to folders labelled with corresponding sum.
- Rename image tiles to include sum as filename prefix.

![alt text](https://github.com/JulieLovitt/TreeCRowNN/blob/main/Treepoint_Tiler/Treepoint_Tiler.jpg)

## [Requirements](https://github.com/JulieLovitt/TreeCRowNN/blob/main/Treepoint_Tiler/requirements.txt)
## Main Module Info:
    
    Parameters
        ----------
        img_path : file path to .tif file
            Location of image to be tiled (ideally 3-band, RGB)
            Image must be 10cm spatial resolution if you plan to use tiles with Tree-CRowNN model
        mask_path : file path
            Location of annotation mask .tif file
            Annotation mask must be binary where 1 = target and 0 = non-target
            Must be georeferenced to, and match spatial resolution of, img_path file  
        NoData : int or float (default -9999)
            Value representing NoData in img_path file 
        out_path : folder path where data will be written
        target : float (default 0.2)
            Value identifying data split (train, test, val)
            E.G. A value of 0.2 represents a split (percentage) of 60/20/20
        tile_size : int (default 128)
            Value identifying the desired dimensions of extracted tiles
            Enter value of -1 to randomly generate patch_size between 64-128pix
        padding : int (default 0)
            Value 0-32 identifying the desired amount of zero padding to be added to tiles
            Enter value of -1 to use random integer generator (0-32)
            E.G. A padding value of 14 should be used when tile_size==100 to generate tiles of 128p x 128p
        
        Returns
        -------
        Image tiles saved as 3-band .png files
      
        Tiles are:
            Split into train, test, and validation datasets 
            Organized in folders labelled with tree counts
            Naming convention = sum_y_x_tilesize_padding.png 
                where: y = row and x = col location in img_path file
        
        THE CODE INCLUDES FILTERS TO AVOID NODATA (ALL 0, ALL -9999) AND INCOMPLETE/NON-SQUARE TILES
        A THOROUGH QA/QC OF GENERATED TILES SHOULD BE COMPLETED PRIOR TO USING FOR MODEL DEV/TRANSFERRING

# TreeCRowNN Inference Module
This code executes model inference with batches on CPU

    Parameters
        ----------
        img_path : file path to .tif file
            Location of image for inference, 3-band (RGB) and 10cm spatial resolution
        out_path : folder path 
            Location where model output will be written 
        site : str
            Name to be given to model output 
        NoData : int/float (default -9999)
            Value of NoData in img file 
        tile_size : int
            Value identifying desired forest stand size 
            E.G. TreeCRowNN accepts tiles with size 128, either use this or padding to achieve correct tile dimensions
        patch_size : int
            Value identifying the desired dimensions of extracted tiles
            E.G. A value of 0 should be used for tiles with dimension 128pix X 128pix
        batch_size : int (default 100)
            Size of batch to be sent to model for inference
        minval : int (default 0)
            minimum data value of input image to be used in normalization
        maxval : int (default 255)
            maximum data value of input image to be used in normalization
        generate_heatmap : bool (default False)
            enter True to generate activation map in addition to FSD map
            if True, processing time per stand will increase ~4x
        normalize_heatmap : str (default none)
            enter "local", "global", or "none"
            if local: heatmap will be normalized per tile (useful to check for border/inter-tile issues)
            if global: heatmap will be normalized for full input image (useful to check for domain trends)
            if none: no normalization will be applied
            
    Returns
        -------
        Array with predictions entered as integers, this array saved to .tif file in out_path location
        Activation heatmaps if requested, PLEASE NOTE: GENERATING HEATMAPS INCREASES PROCESSING TIME ~4x
        Inference processing time in seconds
        
        OUTPUT WILL NOT BE GEOREFERENCED, please use Georeferencing script to transfer metadata for use in a GIS

# [TreeCRowNN Output Georeferencing Module](https://github.com/JulieLovitt/TreeCRowNN/tree/main/TreeCRowNN_Georeferencing)
This code will accept FSD and activation heatmaps for georeferencing to original input RGB image extents

    Parameters
        ----------
        geo_path : file path to .tif file
            Location of image with georeferencing, 3-band (RGB) used as input to inference script
        data_path : file path to .tif file
            Location of image requiring georeferencing, either FSD or activation map
        out_path : file path to .tif file
            File to be written
        tile_size : int
            Spatial resolution of input image (E.G. 0.1 for activation map, 64-128 for FSD) 
        is_FSD : bool (default True)
            enter True to georeference FSD image (coarser spatial resolution)
            enter False to georeference activation map (same spatial resolution as RGB input)
     Returns
        -------
        One georeferenced array saved to .tif file as directed by output_raster settings
        

            
        
