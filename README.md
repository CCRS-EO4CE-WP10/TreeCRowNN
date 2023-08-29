# TreeCRowNN Project
Welcome to the Tree Convolutional Row Neural Network (Tree-CRowNN) Project!   
  
This project supports estimations of forest stand density (FSD) from high-resolution RGB aerial imagery. In this case "forest" density is defined as the count of individual green tree tops that can be identified in 10cm resolution imagery. It does not include standing dead wood, understory/shrubby vegetation, or tree species with diffuse canopy that cannot be easily distinguished from ground at 10cm scale.
  
Imagery used in the initial model development is plane-based 10cm spatial resolution, collected over a mountainous region of interior BC in October 2019.
### The project contains three modules to support: 
- Tile Generation
- Model Inference
- Georeferencing Output
## [License](https://github.com/JulieLovitt/TreeCRowNN/blob/main/LICENSE)

# [Treepoint Tiler Module](https://github.com/JulieLovitt/TreeCRowNN/tree/main/treepoint_tiler)
This module will generate tiles from an image that can then be used in model development or transfer learning.   
It accepts two input: an image you wish to extract tiles from and a complementary binary raster (annotation file). 
  
It will: 
1. Divide the image to train,test,val datasets based on target %.
2. Extract tiles from image and add padding (if desired).
3. Extract tiles from annotation file and compute sum of 1s for each to determine tree counts.
4. Move image tiles to folders labelled with corresponding sum.
5. Rename image tiles to include sum as filename prefix.

![alt text](https://github.com/JulieLovitt/TreeCRowNN/blob/main/treepoint_tiler/Treepoint_Tiler.jpg)

## [Requirements](https://github.com/JulieLovitt/TreeCRowNN/blob/main/treepoint_tiler/requirements.txt)
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

# [TreeCRowNN Inference Module](https://github.com/JulieLovitt/TreeCRowNN/tree/main/inference)
This code executes model inference with batches on CPU. It accepts one input: the 10cm RGB image you wish to run inference on.
  
We provide access to multiple model versions as they are developed. Please note accuracy and settings are different between models. 
Be careful to select the model that best fits your needs. **Models will require validation on new datasets and forest types.** 
The Treepoint Tiler module can be used to generate tiles from new annotated datasets to facilitate transfer learning or fine-tuning.
  
## Tree-CRowNNv1:
The original model as described in our article: *currently in review, link pending*.   
This model generates predictions of FSD at 12.8m resolution and cannot accept padded input.   
  
**Accuracy on Testset:**  
|*n*|Tile Size|Forest Stand Area (m<sup>2</sup>)|True Total Tree Count|Predicted Total Tree Count|diff|MAE|RMSE|R<sup>2</sup>|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|7,457|128|163.84|  126,626 |   |  |2.1|2.7|0.74|
  
**Assessment of Accuracy Across Forest Conditions:**      
|*n*|Forest Condition |MAE| MSE|RMSE|R<sup>2</sup>|
|:---:|:---|:---:|:---:|:---:|:---:|
|20|1: Sparse |2.7|12.0|3.5| - |
|20|2: Immature|1.8|6.0|2.4| - |
|20|3a: Mature, homogenous mix|1.1|3.3|1.8|  - |
|20|3b: Mature, heterogenous mix|2.4|14.7|3.8| - |
|20|4: Undisturbed|2.5|9.8|3.1| - |
|100|ALL|2.1|9.1|3.0|0.87
  
## Tree-CRowNNv2:  
This model leverages zero padding to generate FSD estimates at any scale between 6.4m and 12.8m.  
  
**Accuracy Across Padding Levels and on Combined Dataset (all 128 + all random padding):**  
|*n*|Tile Size |Forest Stand Area (m<sup>2</sup>)| True Total Tree Count| Predicted Total Tree Count| diff| MAE|RMSE|R<sup>2</sup>| 
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|7,905|64|40.96|35,827|27,109|-8,718|1.24|1.62| - |
|7,515|96|92.16|72,341|67,602|-4,739|1.53|2.18| - |
|7,457|128|163.84|126,626|123,580|-3,046|1.84|2.42|0.79|
|14,926|Combo|various|199,610|193,136|-6,474|1.60|2.15|0.88|

  ## [Requirements](https://github.com/JulieLovitt/TreeCRowNN/blob/main/inference/requirements.txt)
  ## Main Module Info:
    Parameters
        ----------
        img_path : file path to .tif file
            Location of image for inference, 3-band (RGB) and 10cm spatial resolution
        out_path : folder path 
            Location where model output will be written 
        site : str
            Name to be given to model output 
        mod : str
            Enter either v1 or v2 to choose TreeCRowNN model 
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

# [TreeCRowNN Output Georeferencing Module](https://github.com/JulieLovitt/TreeCRowNN/tree/main/treecrownn_georef)
This code will accept FSD and activation heatmaps for georeferencing to original input RGB image extents

## [Requirements](https://github.com/JulieLovitt/TreeCRowNN/blob/main/treecrownn_georef/requirements.txt)
## Main Module Info:
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
        One georeferenced array saved to .tif file as directed by out_path settings
        

            
        
