"""
Author: Shi Qiu
Email: shi.qiu@uconn.edu
Date: 2024-05-24
Version: 1.0.0
License: MIT

Description:
This script defines utils of processing the data in the package.

Changelog:
- 1.0.0 (2024-05-24): Initial release.
"""

# pylint: disable=line-too-long
import os
import random
import glob
from pathlib import Path
import re  # regular expression to find image names
import numpy as np
import rasterio
from rasterio import warp
from rasterio.windows import Window
from rasterio.enums import Resampling
import pandas
from skimage.measure import block_reduce, label, regionprops
from skimage.feature import local_binary_pattern
from skimage.morphology import binary_dilation, binary_erosion, reconstruction
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import constant as C
from scipy.ndimage import convolve

# ignore the invalid errors
np.seterr(invalid='ignore') 

# Use non-GUI backend only if running in Jupyter Notebook
# Prevents X server errors when running in a headless environment
def is_running_in_jupyter():
    """Check if the script is running inside a Jupyter Notebook."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False
if is_running_in_jupyter():
    # mpl.use('TkAgg')
    pass
else:
    mpl.use('Agg')


# Functions of index calculationg
def ndvi(red, nir):
    """Normalized Difference Vegetation Index

    Args:
        red (float): Red band
        nir (float): NIR band

    Returns:
        float: ndvi
    """
    # _ndvi = (nir - red) / (nir + red + C.EPS)
    # Efficiently handle division by zero
    _ndvi = np.where((nir + red + C.EPS) == 0, 0, (nir - red) / (nir + red + C.EPS))
    _ndvi[(nir + red) == 0] = 0.01  # fix unnormal pixels
    return _ndvi

def ndsi(green, swir):
    """Normalized Difference Snow Index

    Args:
        green (float): Green band
        swir (float): SWIR1 band

    Returns:
        float: ndsi
    """
    # _ndsi = (green - swir) / (green + swir + C.EPS)
    # Efficiently handle division by zero
    _ndsi = np.where(
        (green + swir + C.EPS) == 0, 0, (green - swir) / (green + swir + C.EPS)
    )
    _ndsi[(green + swir) == 0] = 0.01  # fix unnormal pixels
    return _ndsi

def ndbi(nir, swir):
    """Normalized Difference Build-up Index

    Args:
        nir (float): NIR band
        swir (float): SWIR1 band

    Returns:
        float: ndbi
    """
    # ormalized Difference Build-up Index (NDBI)
    # _ndbi = (swir - nir) / (swir + nir + C.EPS)  # not 0.01 any more because we will identify urban pixel using ndbi more than 0.
    # Efficiently handle division by zero
    _ndbi = np.where((swir + nir + C.EPS) == 0, 0, (swir - nir) / (swir + nir + C.EPS))

    return _ndbi

def hot(blue, red):
    """HOT

    Args:
        blue (float): Blue band
        red (float): Red band

    Returns:
        float: HOT
    """
    return blue - 0.5 * red - 0.08

def whiteness(blue, green, red, satu):
    """Whiteness

    Args:
        blue (float): Blue band
        green (float): Green band
        red (float): Red band
        satu (bool): Radiometric saturation band

    Returns:
        float: whiteness
    """
    visible_mean = (blue + green + red) / 3
    _whiteness = (
        np.absolute(blue - visible_mean)
        + np.absolute(green - visible_mean)
        + np.absolute(red - visible_mean)
    ) / (visible_mean + C.EPS)
    del visible_mean
    _whiteness[satu] = 0
    return _whiteness

def sfdi(blue, swir1, obsmask):
    """Spatial FlooD-filled Index

    Args:
        blue (float): Blue band
        swir1 (float): SWIR1 band
        obsmask (bool): extent of observation

    Returns:
        float: sfdi
    """
    return np.maximum(
        imfill(1.0 - blue, obsmask) - (1.0 - blue),
        imfill(1.0 - swir1, obsmask) - (1.0 - swir1),
    )

def cdi(vre3, wnir, nnir):
    """
    Calculates the Change Detection Index (CDI) for given input bands.

    Args:
        vre3 (ndarray): Array representing the VRE3 band.
        wnir (ndarray): Array representing the WNIR band.
        nnir (ndarray): Array representing the NNIR band.

    Returns:
        ndarray: Array representing the calculated CDI.

    """
    # radius = 3 # same as to 7 in MATLAB's stdfilt(I,nhood)
    # n_points = 8 * radius
    # ratio_8A_8 = pred_layers[predictors.index("wnir"),:,:]/pred_layers[predictors.index("nnir"),:,:] # S2band8./S2band8A;
    # ratio_8A_7 = pred_layers[predictors.index("vre3"),:,:]/pred_layers[predictors.index("nnir"),:,:] # S2band7./S2band8A;
    # # VAR is a rotation invariant measure of local variance
    # var_ratio_8A_8  = local_binary_pattern(np.array(ratio_8A_8*10000*, dtype=np.int16), n_points, radius, 'var')/100000000
    # var_ratio_8A_7  = local_binary_pattern(np.array(ratio_8A_7*10000, dtype=np.int16), n_points, radius, 'var')/100000000

    # VAR is a rotation invariant measure of local variance
    _scale = 10000
    var_ratio_8a_8 = (
        local_binary_pattern(
            _scale * (wnir / (nnir + C.EPS)),
            24,
            3,
            "var",
        )
        / _scale**2
    )  # convert to int from float to avoid the warning from local_binary_pattern It is recommended to use this function with images of integer dtype.
    var_ratio_8a_7 = (
        local_binary_pattern(
            _scale * (vre3 / (nnir + C.EPS)),
            24,
            3,
            "var",
        )
        / _scale**2
    )  # convert to int from float to avoid the warning from local_binary_pattern
    return (var_ratio_8a_7 - var_ratio_8a_8) / (var_ratio_8a_8 + var_ratio_8a_7 + C.EPS)

def variation(nir, radius=5):
    """Spatial variation
        we do not consider edge pixels to save the computing time

    Args:
        nir (2d array): NIR band
        radius (int, optional): Radius of kernel. Defaults to 5.

    Returns:
        float: the variation image
    """
    _scale = 10000
    # settings for LBP
    n_points = 8 * radius
    # to convert to int from float to avoid the warning from local_binary_pattern
    return (
        local_binary_pattern(
            np.array(nir * _scale, dtype=np.int32),
            n_points,
            radius,
            "var",
        )
        / _scale**2
    )  # convert to int from float to avoid the warning from local_binary_pattern

#%% Functions of processing the raster data
def warp2like(src, like, des=None):
    """
    Warp the imagery to match the extent and resolution of the like-data.

    Args:
        src (str): The filepath of the source data.
        like (str or dict): Option 1: The file path of the like data.
                            Option 2: The profile of the like data.
        des (str, optional): The filepath to save the warped data. If not provided, the function returns the warped data as a 2D or 3D array.

    Returns:
        numpy.ndarray: The warped data as a 2D or 3D array, depending on the number of bands in the source data.

    """
    with rasterio.open(src, "r") as src_mask:  # obtain the resource dataset
        des_profile = src_mask.meta.copy()

        # update the geo-relatives from the like-imagery
        if isinstance(like, str):
            # update the profile same as to the like-geotiff
            with rasterio.open(like, "r") as src_like:
                des_profile.update(
                    {
                        "crs": src_like.crs,
                        "transform": src_like.transform,
                        "width": src_like.width,
                        "height": src_like.height,
                    }
                )
        else:
            # move forward with the profile defined ahead of time
            try:
                des_profile.update(
                    {
                        "crs": like["crs"],
                        "transform": like["transform"],
                        "width": like["width"],
                        "height": like["height"],
                    }
                )
            except:
                des_profile.update(
                    {
                        "crs": like.crs,
                        "transform": like.transform,
                        "width": like.width,
                        "height": like.height,
                    }
                )
        # reproject and warp to the like-imagery
        if des is not None:
            # save as local file # warp as the destination geotif
            with rasterio.open(des, "w", **des_profile) as dst_mask:
                for i in range(1, src_mask.count + 1):
                    warp.reproject(
                        source=rasterio.band(src_mask, i),
                        destination=rasterio.band(dst_mask, i),
                        src_transform=src_mask.transform,
                        src_crs=src_mask.crs,
                        dst_transform=des_profile["transform"],
                        dst_crs=des_profile["crs"],
                        resampling=warp.Resampling.nearest,
                        dst_nodata=src_mask.nodata,
                    )
        else:
            # return the 2d-array
            if src_mask.count == 1:
                # copy the processing to avoid converting 3d to 2d array at the last
                dst_mask = np.zeros(
                    (des_profile["height"], des_profile["width"]),
                    dtype=src_mask.meta["dtype"],
                )
                warp.reproject(
                    source=rasterio.band(src_mask, 1),  # the 1st band
                    destination=dst_mask[:, :],
                    src_transform=src_mask.transform,
                    src_crs=src_mask.crs,
                    dst_transform=des_profile["transform"],
                    dst_crs=des_profile["crs"],
                    resampling=warp.Resampling.nearest,
                    dst_nodata=src_mask.nodata,
                )
            elif src_mask.count > 1:
                # also support multiple bands
                dst_mask = np.zeros(
                    (des_profile["height"], des_profile["width"], src_mask.count),
                    dtype=src_mask.meta["dtype"],
                )
                for i in range(1, src_mask.count + 1):
                    warp.reproject(
                        source=rasterio.band(src_mask, i),
                        destination=dst_mask[:, :, i - 1],
                        src_transform=src_mask.transform,
                        src_crs=src_mask.crs,
                        dst_transform=des_profile["transform"],
                        dst_crs=des_profile["crs"],
                        resampling=warp.Resampling.nearest,
                        dst_nodata=des_profile["nodata"],
                    )
            return dst_mask

def gen_dem(profile, des=None):
    """
    Generate a digital elevation model (DEM) using gtopo30 data.

    Args:
        profile (dict): The profile of the input raster.
        des (str, optional): The destination path for the generated DEM. Defaults to None.

    Returns:
        numpy.ndarray: The generated DEM as a NumPy array.
    """
    if C.MSG_FULL:
        print(">>> loading dem from gtopo30")
    path_dem = os.path.join(Path(__file__).parent.parent, "data", "global_gt30.tif")
    return warp2like(src=path_dem, like=profile, des=des)

def gen_slope(profile, des=None):
    """
    Generate a slope raster based on the given profile.
    This function loads a global slope dataset (gtopo30-slope) and warps it to match the given profile.
    Args:
        profile (dict): The profile (metadata) of the target raster to match.
        des (str, optional): The destination path to save the warped raster. If None, the result is returned as an array.
    Returns:
        numpy.ndarray or None: The warped slope raster as a numpy array if `des` is None, otherwise None.
    """

    if C.MSG_FULL:
        print(">>> loading gtopo30-slope")
    return warp2like(src=os.path.join(Path(__file__).parent.parent, "data", "global_gt30_slope.tif"), like=profile, des=des)/100

def gen_aspect(profile, des=None):
    """
    Generates an aspect raster that matches the given profile.
    Parameters:
    profile (dict): The profile dictionary containing metadata for the raster.
    des (str, optional): The destination path for the output raster. Defaults to None.
    Returns:
    ndarray: The aspect raster aligned with the given profile.
    """

    if C.MSG_FULL:
        print(">>> loading gtopo30-aspect")
    return warp2like(src=os.path.join(Path(__file__).parent.parent, "data", "global_gt30_aspect.tif"), like=profile, des=des)/100
    

def gen_gswo(profile, des=None):
    """
    Generate a global surface water mask.

    Args:
        profile (dict): The profile of the input raster.
        des (str, optional): The destination path for the generated surface water mask. Defaults to None.

    Returns:
        numpy.ndarray: The generated surface water mask.

    Raises:
        None

    Notes:
        This function only works on the global layer in the package, which was produced by create_global_gswo150.py.
        The image should be within the global water layer at coordinates 78, -59.
    """
    if C.MSG_FULL:
        print(">>> loading gswo")
    # Note: this function only works on the gobal layer in the package, which was produced by create_global_gswo150.py
    # check the image is within the global water layer, 78, -59
    path_gswo = os.path.join(Path(__file__).parent.parent, "data", "global_gswo150.tif")
    swo = warp2like(src=path_gswo, like=profile, des=des)
    if swo is not None:
        swo[swo == 255] = 100  # 255 is 100% ocean.
    return swo

def topo_correct_scs(band_ori1, band_ori2, sun_elevation_deg, sun_azimuth_deg, slope_data, aspect_data):
    """
    Applies topographic correction SCS to the input band data.
    The SCS correction is equivalent to projecting the sunlit canopy from the sloped surface to the horizontal surface in the direction of illumination (Gu, D. et al., 1998).

    Parameters:
        band_ori (numpy.ndarray): Original band data (2D array).
        sun_elevation_deg (float): Solar elevation angle in degrees (number).
        sun_azimuth_deg (float): Solar azimuth angle in degrees (number).
        slope_data (numpy.ndarray): Slope data in degrees (2D array).
        aspect_data (numpy.ndarray): Aspect data in degrees (2D array).

    Returns:
        numpy.ndarray: Corrected band data.
    """

    # Convert angles to radians
    # Convert solar elevation angle to solar zenith angle
    sun_zenith_rad = np.radians(90 - sun_elevation_deg)
    sun_zenith_cos = np.cos(sun_zenith_rad)
    sun_zenith_sin = np.sin(sun_zenith_rad)
    del sun_zenith_rad

    slope_rad = np.radians(slope_data)
    aspect_rad = np.radians(sun_azimuth_deg - aspect_data)

    # Calculate cos_sita: the cosine of the angle between sun and surface normal
    cos_sita = (sun_zenith_cos * np.cos(slope_rad) +
                sun_zenith_sin * np.sin(slope_rad) * np.cos(aspect_rad))

    # Create a mask to check if the correction should be applied Ref. Tan et al. RSE (2013)
    cor_mask = np.abs(cos_sita - sun_zenith_cos) > 0.05
    if np.any(cor_mask):
        # Apply the correction to the band only where cor_mask is True
        band_corrected1 = band_ori1.copy()
        band_corrected1[cor_mask] = (
                band_ori1[cor_mask] * 
                (np.cos(slope_rad[cor_mask]) * sun_zenith_cos) / cos_sita[cor_mask]
            )
        band_corrected2 = band_ori2.copy()
        band_corrected2[cor_mask] = (
                band_ori2[cor_mask] * 
                (np.cos(slope_rad[cor_mask]) * sun_zenith_cos) / cos_sita[cor_mask]
            )
        return band_corrected1, band_corrected2
    else:
        return band_ori1, band_ori2

def erode(mask, radius=3):
    """
    Erodes the input mask by a given radius.

    Args:
        mask (ndarray): Binary mask to be eroded.
        radius (int, optional): Radius of the erosion. Defaults to 3.

    Returns:
        ndarray: Eroded mask.
    """
    return binary_erosion(
        mask, footprint=np.ones((2 * radius + 1, 2 * radius + 1)), out=None
    )

def dilate(mask, radius=3):
    """
    Dilates the input mask by a given radius.

    Args:
        mask (ndarray): Binary mask to be dilated.
        radius (int, optional): Radius of the dilation. Defaults to 3.

    Returns:
        ndarray: Dilated mask.
    """
    #return binary_dilation(
    #    mask, footprint=np.ones((2 * radius + 1, 2 * radius + 1)), out=None
    #)
    max_step = 10 # too large radius will cost too much time and memory
    if radius <= max_step:
        return binary_dilation(
            mask, footprint=np.ones((2 * radius + 1, 2 * radius + 1)), out=None
        )
    else:
        # use a larger kernel to speed up the process
        i = 1
        while i < radius:
            current_step = min(max_step, radius - i + 1)
            mask = binary_dilation(
                mask, footprint=np.ones((2 * current_step + 1, 2 * current_step + 1)), out=None
            )
            i += max_step
        return mask

def imfill(data, obsmask, fill_value=None):
    """
    Fills holes in the grayscale 2d data using morphological reconstruction by erosion,
    where a hole is defined as an area of dark pixels surrounded by lighter pixels.
    Erosion expands the minimal values of the seed image until it encounters a mask image (maximal).
    Thus, the seed image and mask image represent the maximum and minimum possible values of the reconstructed image.
    Note more than 20 seconds will be cost for processing a Landsat imagery
    Also see the example at https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html

    This is same as the function of Matlab's imfill, see https://www.mathworks.com/help/images/ref/imfill.html
    This version has been verfied as to ' imfill(image, 8)' in MATLAB.
    NOTE 8-conn is faster than 4-conn in MATLAB.
    """
    # np.seterr(invalid='warn') # not ignore the invalid errors
    max_val = data[obsmask].max()  # the maximum value within the image extent
    if fill_value is None:
        fill_value = max_val
    # data[~obsmask] = fill_value  # fill the pixles out of the image extent
    # fill the nan values by the nearest values from row, and from column when the observation mask is not full
    # if False in obsmask:
    #     data[~obsmask] = np.nan  # fill the pixles out of the image extent
    #     data = fill_nan_wise(data) # fill the nan values by the nearest values from row, and from column
    data[~obsmask] = fill_value
    seed = np.copy(data)
    seed[1:-1, 1:-1] = max_val

    # # erode the observation extent, of which boundary pixels will be remained to star the erosion process
    # obsmask_eroded = binary_erosion(obsmask, footprint=np.ones((3, 3)), out=None)
    # # cut off the boundary pixels. This process is able to ensure the starting points alway exists.
    # obsmask_eroded[0, :] = False
    # obsmask_eroded[-1, :] = False
    # obsmask_eroded[:, 0] = False
    # obsmask_eroded[:, -1] = False

    # # remain the boundary pixel of the image as the original values of the image. These border pixels will be the starting points for the erosion process.
    # # create the seed image, where the minima represent the starting points for erosion.
    # # we initialize the seed image to the maximum value of the original image.
    # seed = np.copy(data)
    # seed[obsmask_eroded] = max_val
    # del obsmask_eroded

    # fill the data
    # a “seed” image, which specifies the values that spread, and a "data" (or "mask") image, which gives the maximum allowed value at each pixel
    
    # Check the assertion
    # assert np.all(seed <= data), "Error: seed must be <= data for erosion."
    # seed = np.minimum(seed, data)
    data_filled = reconstruction(
        seed, data, method="erosion"
    )  # a 3x3 square, of which return is float64
    # data_filled = data_filled.astype(
    #    np.float32
    # )  # covnert to the scalar that we want to use
    return data_filled

#%% Functions of reading and processing the reference mask


def read_reference_mask(des, dataset="l8biome", shape=None):
    """
    Read and process a reference mask based on the dataset.

    Args:
        des (str): The path to the reference mask file.
        dataset (str, optional): The dataset type. Defaults to "l8biome".
        shape (list, optional): The target shape for resampling. Defaults to None.

    Returns:
        numpy.ndarray: The processed reference mask, with Fmask labels, clear, shadow, cloud, and filled

    Raises:
        FileNotFoundError: If the reference mask file is not found.

    """

    if dataset.lower() == "l8biome":
        # Value	Interpretation
        # 0	       Fill
        # 64	   Cloud Shadow
        # 128	   Clear
        # 192	   Thin Cloud
        # 255	   Cloud

        des_folder = Path(des)
        des_name = Path(des_folder).stem
        # read the manual mask
        with rasterio.open(
            os.path.join(des_folder, des_name + "_FIXED_MASK.TIF")
        ) as src:
            mask_manul = src.read(1)
        mask_manul_pre = mask_manul.copy()
        mask_manul[mask_manul_pre == 128] = C.LABEL_CLEAR
        mask_manul[mask_manul_pre >= 192] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre == 64] = C.LABEL_SHADOW
        mask_manul[mask_manul_pre == 0] = C.LABEL_FILL

    if dataset.lower() == "l8sparcs":
        # Value	Interpretation
        # 0    Shadow
        # 1    Shadow over Water
        # 2    Water
        # 3    Snow
        # 4    Land
        # 5    Cloud
        # 6    Flooded

        des_folder = Path(des)
        des_name = Path(des_folder).stem
        # read the manual mask
        with rasterio.open(os.path.join(des_folder, des_name + "_mask.png")) as src:
            mask_manul = src.read(1)
        mask_manul_pre = mask_manul.copy()
        mask_manul[mask_manul_pre >= 0] = C.LABEL_CLEAR
        mask_manul[mask_manul_pre == 5] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre <= 1] = C.LABEL_SHADOW

    if dataset.lower() == "l895cloud":  # Value	Interpretation
        # 0      Non-cloud
        # 1      Cloud
        # other  Filled

        des_folder = Path(des)
        des_name = Path(des_folder).stem
        # read the manual mask
        with rasterio.open(os.path.join(des_folder, des_name + "_MASK.TIF")) as src:
            mask_manul = src.read(1)

        mask_manul_pre = mask_manul.copy()
        mask_manul[mask_manul_pre == 0] = C.LABEL_CLEAR
        mask_manul[mask_manul_pre == 1] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre >= 2] = C.LABEL_FILL  # since it is uint8, we use >=

    if dataset.upper() == "S2WHUCDPLUS":
        # see https://github.com/Neooolee/WHUS2-CD
        # (0, 128 and 255 are for nodata, clear and cloud pixels, respectively)

        des_folder = Path(des)
        des_name = Path(des_folder).stem
        # read the manual mask
        with rasterio.open(
            os.path.join(des_folder, "REFERENCE", des_name + "_Mask.tif")
        ) as src:
            mask_manul = src.read(1)

        # shape are different, we need to resample
        if shape is not None:
            if (
                shape[0] < src.shape[0]
            ):  # when target shape > the raw reference layer's, aggregation will be used
                # majortity vote, and then at the priority squeue: cloud > clear > fill
                mask_manul = block_reduce(
                    mask_manul, block_size=int(src.shape[0] / shape[0]), func=np.max
                )

        mask_manul_pre = mask_manul.copy()
        mask_manul[mask_manul_pre == 128] = C.LABEL_CLEAR
        mask_manul[mask_manul_pre == 255] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre == 0] = C.LABEL_FILL

    if dataset.upper() == "S2ALCD":
        # see https://zenodo.org/records/1460961
        # 0: no_data.
        # 1: not used.
        # 2: low clouds.
        # 3: high clouds.
        # 4: clouds shadows.
        # 5: land.
        # 6: water.
        # 7: snow.

        # which is at 60 meters

        des_folder = Path(des)
        des_name = Path(des_folder).stem
        # read the manual mask
        with rasterio.open(
            os.path.join(
                des_folder, "REFERENCE", "Classification", "classification_map.tif"
            )
        ) as src:
            # shape are different, we need to resample
            if shape is not None:
                if (
                    shape[0] > src.shape[0]
                ):  # when target shape < the raw reference layer's, neariest will be used
                    mask_manul = src.read(
                        1, out_shape=shape, resampling=Resampling.nearest
                    )  # only one band, and nearest is used to resample to smaller resolution

        mask_manul_pre = mask_manul.copy()
        mask_manul[mask_manul_pre >= 5] = C.LABEL_CLEAR
        mask_manul[mask_manul_pre == 2] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre == 3] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre == 4] = C.LABEL_SHADOW
        mask_manul[mask_manul_pre == 0] = C.LABEL_FILL

    if dataset.upper() == "S2IRIS":
        # Each mask are a 1022-by-1022-by-3 numpy array, with boolean one-hot encoding (each pixel has exactly one True value across the last
        # dimension). The class order in the last dimension is: CLEAR, CLOUD, CLOUD_SHADOW
        des_folder = Path(des)
        des_name = Path(des_folder).stem
        mask_manul = np.load(os.path.join(des_folder, "REFERENCE", des_name + ".npy"))
        mask_manul_pre = mask_manul.copy()
        mask_manul = np.full((mask_manul.shape[0], mask_manul.shape[1]), 0)
        mask_manul[mask_manul_pre[:, :, 0]] = C.LABEL_CLEAR
        mask_manul[mask_manul_pre[:, :, 1]] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre[:, :, 2]] = C.LABEL_SHADOW
    # S2CloudSEN12Plus 
    if dataset.upper() == "S2CLOUDSEN12PLUS":
        des_folder = Path(des)
        des_name = Path(des_folder).stem
        # read the manual mask with rasterio
        with rasterio.open(
            os.path.join(des_folder, des_name + "_HighCM.tif")
        ) as src:
            mask_manul = src.read(1)
        mask_manul_pre = mask_manul.copy()
        mask_manul = np.full((mask_manul.shape[0], mask_manul.shape[1]), 0)
        mask_manul[mask_manul_pre == 0] = C.LABEL_CLEAR # clear
        mask_manul[mask_manul_pre == 1] = C.LABEL_CLOUD # thick cloud
        mask_manul[mask_manul_pre == 2] = C.LABEL_CLOUD # thin cloud
        mask_manul[mask_manul_pre == 3] = C.LABEL_SHADOW # cloud shadow
 
    if dataset.upper() == "S2FMASK":
        # followed the Fmask labels
        # 0 clear land
        # 1 clear water
        # 2 cloud shadow [but not used in S2FMASK]
        # 3 snow
        # 4 cloud
        # 255 fill
        des_folder = Path(des)
        des_name = Path(des_folder).stem
        # read the manual mask
        with rasterio.open(
            os.path.join(des_folder, "REFERENCE", des_name + "_MASK.tif")
        ) as src:
            mask_manul = src.read(1)

        # shape are different, we need to resample
        if shape is not None:
            if (
                shape[0] < src.shape[0]
            ):  # when target shape > the raw reference layer's, aggregation will be used
                # majortity vote, and then at the priority squeue: cloud > clear > fill
                mask_manul = block_reduce(
                    mask_manul, block_size=int(src.shape[0] / shape[0]), func=np.max
                )

        mask_manul_pre = mask_manul.copy()
        mask_manul[mask_manul_pre <=1]  = C.LABEL_CLEAR
        mask_manul[mask_manul_pre ==3]  = C.LABEL_CLEAR
        mask_manul[mask_manul_pre == 4] = C.LABEL_CLOUD
        mask_manul[mask_manul_pre == 255] = C.LABEL_FILL
    return mask_manul

def collect_sample_pixel(
    data,
    bands,
    reference: np.ndarray,
    landcover=None,
    label_cloud=None,
    label_fill=None,
    number=1000,
    cloud_area=True,
    scene_prct=True,
):
    """
    Collects sample pixels from the input data and reference image.

    Args:
        data (np.ndarray): The input datacube.
        bands (list): The list of band names.
        reference (np.ndarray): The reference image.
        landcover (np.ndarray, optional): The landcover image. Defaults to None.
        label_cloud (int, optional): The label for cloud pixels. Defaults to None.
        label_fill (int, optional): The label for fill pixels. Defaults to None.
        number (int or float, optional): The number of sample pixels to collect. Defaults to 1000.
        cloud_area (bool, optional): Whether to include cloud size in the output. Defaults to True.
        scene_prct (bool, optional): Whether to include cloud cover percentage of the scene in the output. Defaults to True.

    Returns:
        pandas.DataFrame: The dataset containing the collected sample pixels.
    """

    if label_cloud is None:
        label_cloud = C.LABEL_CLOUD
    if label_fill is None:
        label_fill = C.LABEL_FILL

    # Static seed for reproductivity
    random.seed(C.RANDOM_SEED)
    # select the sample pixels randomly
    sample_pixels_row, sample_pixels_col = np.where(reference != label_fill)
    # number or ratio
    if number < 1:
        number = number * len(sample_pixels_row)

    ids_selected = random.sample(
        range(0, len(sample_pixels_row)), np.min([len(sample_pixels_row), int(number)])
    )  # Totally random
    sample_pixels_row, sample_pixels_col = (
        sample_pixels_row[ids_selected],
        sample_pixels_col[ids_selected],
    )
    del ids_selected

    # get the datacube and label
    x_inputs = data[:, sample_pixels_row, sample_pixels_col]

    x_inputs = np.swapaxes(x_inputs, 0, 1)  # rotate it as to
    y_input = reference[sample_pixels_row, sample_pixels_col]

    df_data = np.expand_dims(sample_pixels_row, 1)
    df_data = np.append(df_data, np.expand_dims(sample_pixels_col, 1), 1)
    df_data = np.append(df_data, x_inputs, 1)
    df_data = np.append(df_data, np.expand_dims(y_input, 1), 1)
    if landcover is not None:
        sample_cover = landcover[sample_pixels_row, sample_pixels_col]
        df_data = np.append(df_data, np.expand_dims(sample_cover, 1), 1)
    if cloud_area:  # make sure to include the cloud size
        # create the cloud object for the cloud mask, and know the cloud object's size in pixels, 30m landsat and 20m sentinel-2
        cloud_labels = label(reference == label_cloud, background=0)
        cloud_props = regionprops(cloud_labels, reference == label_cloud)
        # For each label, and assign the number of pixels to the cloud object into the images
        for index in range(0, cloud_labels.max()):
            cloud_coords = cloud_props[index].coords
            cloud_num_pixels = cloud_props[index].area
            cloud_labels[cloud_coords[:, 0], cloud_coords[:, 1]] = cloud_num_pixels
        cloud_size = cloud_labels[sample_pixels_row, sample_pixels_col]
        df_data = np.append(df_data, np.expand_dims(cloud_size, 1), 1)
        # cloud cover percentage of the scene
    if scene_prct:
        cloud_scene_percent = np.count_nonzero(
            reference == label_cloud
        ) / np.count_nonzero(reference != label_fill)
        df_data = np.append(
            df_data, np.expand_dims(np.full(len(y_input), cloud_scene_percent), 1), 1
        )  # share the same value for each imagery

    # set as dataframe to return
    df = pandas.DataFrame(data=df_data)
    # set column names
    predictors = bands.copy()  # copy the bands as the predictors
    predictors.insert(0, "image_row")
    predictors.insert(0, "image_col")
    predictors.append("label")
    if landcover is not None:
        predictors.append("landcover")
    if cloud_area:  # make sure to include the cloud size
        predictors.append("cloud_size")
    if scene_prct:
        predictors.append("cloud_scene_percent")
    df.columns = predictors
    # return the dataset for training models
    return df

def init_patch_offanchors(obsmask, size=512, stride=488, shift=True):
    """
    Initialize patch offset anchors for sampling square patches from an observation mask.
    This function identifies anchor positions for extracting patches from a binary observation mask,
    starting from the median location of valid pixels and moving outward in four directions (up, down,
    left, right) with a specified stride. It ensures that each patch is within image bounds and attempts
    to center valid pixels within each patch when possible.
    Parameters
    ----------
    obsmask : np.ndarray
        2D binary array (mask) indicating valid pixels (nonzero values).
    size : int, optional
        Size of the square patch to extract (must be even), by default 512.
    stride : int, optional
        Step size for moving the patch anchor in each direction, by default 488.
    Returns
    -------
    offanchors : list of tuples
        List of tuples containing offsets for each patch:
        (row_offset, col_offset, row_default, col_default)
        where:
            - row_offset, col_offset: possibly shifted offsets to better center valid pixels
            - row_default, col_default: original offsets before shifting
    Notes
    -----
    - Patches that do not contain any valid pixels are skipped.
    - If a patch contains filled (invalid) pixels, the anchor is shifted to better center valid pixels.
    - The function ensures that all returned offsets are within the image bounds.
    """

    # Initialize the offsets for sampling patches from the observation mask.
    # size must be even number, like 256

    # starting from central pixel, and then move to the four directions
    # if the image is not full, we need to fill the nan values by the nearest
    # define the anchor, using the median location of the valid pixels
    height, width = obsmask.shape

    # Get row and column indices of valid pixels
    rows, cols = np.where(obsmask)
    # Compute the median of valid positions, starting from the center of the image
    off_anchor_row_seed = int(np.median(rows)) - size // 2
    off_anchor_col_seed = int(np.median(cols)) - size // 2

    # Ensure the anchor is within the image bounds
    off_anchor_row_seed = np.clip(off_anchor_row_seed, 0, height - size)
    off_anchor_col_seed = np.clip(off_anchor_col_seed, 0, width - size)
    
    # Get center positions from anchor
    if shift:
        off_anchor_rows = np.concatenate([np.arange(off_anchor_row_seed, - size, -stride, dtype=int)[::-1], np.arange(off_anchor_row_seed + stride, height, stride, dtype=int)])
        off_anchor_cols = np.concatenate([np.arange(off_anchor_col_seed, - size, -stride, dtype=int)[::-1], np.arange(off_anchor_col_seed + stride, width, stride, dtype=int)])
    else: # do not take the patches which poentially cover the extent out of the imagery # -1 does not be included
        off_anchor_rows = np.concatenate([np.arange(off_anchor_row_seed, -1, -stride, dtype=int)[::-1], np.arange(off_anchor_row_seed + stride, height - stride, stride, dtype=int)])
        off_anchor_cols = np.concatenate([np.arange(off_anchor_col_seed, -1, -stride, dtype=int)[::-1], np.arange(off_anchor_col_seed + stride, width - stride, stride, dtype=int)])
 
    # index of recording
    offanchors = []
    for r_off_i in off_anchor_rows:
        for c_off_i in off_anchor_cols:
            # Ensure the center is within the image extent, which are not considered as the shifted offsets, 
            r_off_default = np.clip(r_off_i, 0, height - size) # r_center - halfsize is the starting row offset
            c_off_default = np.clip(c_off_i, 0, width - size) # c_center - halfsize is the starting column offset
            # Check if the patch is valid
            obs_patch = obsmask[r_off_default:r_off_default + size, c_off_default:c_off_default + size]
            # If the patch is not valid, skip it
            if not obs_patch.any():
                continue
            r_off_shift = r_off_default.copy()  # r_off_shift is the row offset for the patch
            c_off_shift = c_off_default.copy()  # c_off_shift is the column offset for the patch

            # adjust the offset only when the patch includes filled pixels
            if not np.all(obs_patch): 
                # Optionally shift to center valid pixels
                valid_rows, _ = np.where(obs_patch)
                min_r, max_r = valid_rows.min(), valid_rows.max()
                if min_r > 0: # filled pixels are at the top
                    r_off_shift = np.clip(r_off_shift + min_r, 0, height - size)
                elif max_r < size - 1: # filled pixels are at the bottom
                    r_off_shift = np.clip(r_off_shift - (size - max_r -1), 0, height - size)

                obs_patch = obsmask[r_off_shift:r_off_shift + size, c_off_shift:c_off_shift + size]
                _, valid_cols = np.where(obs_patch)
                min_c, max_c = valid_cols.min(), valid_cols.max()
                if min_c > 0: # filled pixels are at the left
                    c_off_shift = np.clip(c_off_shift + min_c, 0, width - size)
                elif max_c < size - 1: # filled pixels are at the right
                    c_off_shift = np.clip(c_off_shift - (size - max_c - 1), 0, width - size)

            # record the offset anchors
            offanchors.append((r_off_shift, c_off_shift, r_off_default, c_off_default))
    
    # unique the offsets, as to avoid the duplicate patches
    offanchors = list(set(offanchors))  # remove duplicates
    offanchors.sort()  # sort the offsets for consistency
    # if shift is True, we will shift the offsets to center the valid pixels in the patch        
    
    # for the dataset, we do not adjust the boundary patches because of the particular coverage of the dataset
    if not shift:
        # (r_off_shift, c_off_shift, r_off_default, c_off_default), select the items when r_off_shift == r_off_default and c_off_shift == c_off_default
        offanchors = [item for item in offanchors if item[0] == item[2] and item[1] == item[3]]
    return offanchors

def collect_sample_patch(
    dataset,
    image_name,
    image_profile,
    image_profilefull,
    data,
    reference,
    obsmask,
    des,
    size=512,
    stride=488,
    shift=True,
    dformat="tif",
):
    """
    Collects sample patches from an image dataset.

    Args:
        image (Landsat|Sentinel2): The name of the image.
        dataset (str): The name of the dataset.
        reference (np.ndarray): The reference array.
        des (str): The destination directory.
        size (int, optional): The size of the patch. Defaults to 512.
        stride (int, optional): The stride between patches. Defaults to 488.
        shift (bool, optional): Whether to shift patches at the border to cover the valid pixels as much as possible. Defaults to True.
    """
    offanchors = init_patch_offanchors(obsmask, size=size, stride=stride, shift=shift)

    if len(offanchors) == 0:
        print(">>> No valid patches found in the observation mask.")
        return
    # create images and labels' folder
    dir_images = os.path.join(des, "images")
    dir_labels = os.path.join(des, "labels")
    os.makedirs(dir_images, exist_ok=True)
    os.makedirs(dir_labels, exist_ok=True)
    
    # convert to filled images as to 255, both of the two datasets are lack of fill labels
    reference[obsmask == False] = C.LABEL_FILL  # fill the pixels that are not observed with 255
    
    # expand the 1st dimension, as to saving later as .geotif 
    for r_off, c_off, r_off_default , c_off_default in offanchors:
        chipname = (
            f"{dataset.upper()}_{image_name.upper()}_{r_off:05}_{c_off:05}"
        )
        label_patch = reference[r_off : r_off + size, c_off : c_off + size]
        
        # skip the full valid patch with adjusted offsets, since we do not want augment the dulicate patches
        if shift and ((r_off_default != r_off) or (c_off_default != c_off)):
            # if the patch is not valid, skip it
            if not np.any(label_patch == C.LABEL_FILL):
                print(f">>> skip the full valid patch {chipname} with adjusted offsets")
                continue
        
        image_patch = data.data[:, r_off : r_off + size, c_off : c_off + size]
        
        # save it locally
        if dformat.lower() == "npy":
            # Save data to a .npy file without compression
            if not os.path.isfile(os.path.join(dir_images, chipname + ".npy")):
                np.save(os.path.join(dir_images, chipname + "_part.npy"), image_patch)
                os.rename(os.path.join(dir_images, chipname + "_part.npy"), os.path.join(dir_images, chipname + ".npy"))
            if not os.path.isfile(os.path.join(dir_labels, chipname + ".npy")):
                np.save(os.path.join(dir_labels, chipname + "_part.npy"), label_patch.astype(np.uint8))
                os.rename(os.path.join(dir_labels, chipname + "_part.npy"), os.path.join(dir_labels, chipname + ".npy"))
        elif dformat.lower() == "tif":
            # update the profile of geotiff
            transform = image_profilefull.window_transform(Window(
                c_off, r_off, size, size
            ))
            profile = image_profile.copy()
            profile.update(
                {
                    "driver": "GTiff",  # JPG2 does not support some certain data type
                    "nodata": None,  # setup no data as None!
                    "height": size,
                    "width": size,
                    "count": image_patch.shape[0],
                    "dtype": np.float32,
                    "transform": transform,
                }
            )
            if not os.path.isfile(os.path.join(dir_images, chipname + ".tif")):
                with rasterio.open(os.path.join(dir_images, chipname + "_part.tif"), "w+", **profile) as dst:
                    dst.write(image_patch)
                os.rename(os.path.join(dir_images, chipname + "_part.tif"), os.path.join(dir_images, chipname + ".tif"))
            if not os.path.isfile(os.path.join(dir_labels, chipname + ".tif")):
                profile.update({"count": 1, "dtype": np.uint8}) # save label with only one band
                with rasterio.open(os.path.join(dir_labels, chipname + "_part.tif"), "w+", **profile) as dst:
                    dst.write(label_patch[np.newaxis]) # expand dims to 1st dimension to save a single band
                os.rename(os.path.join(dir_labels, chipname + "_part.tif"), os.path.join(dir_labels, chipname + ".tif"))
        # msg
        print(f">>> created images and labels for {chipname}")

def collect_sample_patch2(
    dataset,
    image_name,
    image_profile,
    image_profilefull,
    data,
    reference,
    des,
    size=512,
    stride=488,
    append_end=True,
    dformat="tif",
):
    """
    Collects sample patches from an image dataset.

    Args:
        image (Landsat|Sentinel2): The name of the image.
        dataset (str): The name of the dataset.
        reference (np.ndarray): The reference array.
        des (str): The destination directory.
        size (int, optional): The size of the patch. Defaults to 512.
        stride (int, optional): The stride between patches. Defaults to 488.
        append_end (bool, optional): Whether to append patches at the border. Defaults to True.
    """
    # expand the 1st dimension, as to saving later as .geotif
    reference = np.expand_dims(reference, 0)
    # index of recording
    i = 0
    # Change the r c off
    rows_off = np.arange(0, data.data.shape[1] - size + 1, stride)
    cols_off = np.arange(0, data.data.shape[2] - size + 1, stride)

    # fetch the last chips at the border
    if append_end:
        rows_off = np.unique(np.append(rows_off, data.data.shape[1] - size))
        cols_off = np.unique(np.append(cols_off, data.data.shape[2] - size))

    for r_off in rows_off:
        for c_off in cols_off:
            # setup the sub image
            win = Window(
                c_off, r_off, size, size
            )  # (column_offset, row_offset, width, height)
            datacube_label = reference[:, r_off : r_off + size, c_off : c_off + size]

            # exclude pure dark chips
            if np.count_nonzero(datacube_label < C.LABEL_FILL) == 0:
                continue

            datacube_image = data.data[:, r_off : r_off + size, c_off : c_off + size]

            # update the profile of geotiff
            transform = image_profilefull.window_transform(win)
            profile = image_profile.copy()
            profile.update(
                {
                    "driver": "GTiff",  # JPG2 does not support some certain data type
                    "nodata": None,  # setup no data as None!
                    "height": size,
                    "width": size,
                    "count": datacube_image.shape[0],
                    "dtype": np.float32,
                    "transform": transform,
                }
            )

            # create images and labels' folder
            dir_images = os.path.join(des, "images")
            dir_labels = os.path.join(des, "labels")
            if not os.path.isdir(dir_images):
                os.makedirs(dir_images)
            if not os.path.isdir(dir_labels):
                os.makedirs(dir_labels)
            i = i + 1
            # chipname = "{0:09}_{1:05}_{2:05}_{3}_{4}.tif".format(i, r_off, c_off, dataset, IMAGE.name)
            chipname = (
                f"{dataset.upper()}_{image_name.upper()}_{r_off:05}_{c_off:05}"
            )
            # chipname = "{0}_{1}_{2:05}_{3:05}.tif".format(dataset.upper(), image_name.upper(), r_off, c_off, )

            # save it locally
            if dformat.lower() == "tif":
                chip_path = os.path.join(dir_images, chipname + ".tif")
                with rasterio.open(chip_path, "w+", **profile) as dst:
                    dst.write(datacube_image)

                # save label
                profile.update({"count": datacube_label.shape[0], "dtype": np.uint8})
                chip_path = os.path.join(dir_labels, chipname + ".tif")
                with rasterio.open(chip_path, "w+", **profile) as dst:
                    dst.write(datacube_label)
            elif dformat.lower() == "npy":
                # Save data to a .npy file without compression
                chip_path = os.path.join(dir_images, chipname + ".npy")
                np.save(chip_path, datacube_image)
                chip_path = os.path.join(dir_labels, chipname + ".npy")
                # remove the 1st dimension
                datacube_label = np.squeeze(datacube_label, axis=0)
                np.save(chip_path, datacube_label)

            # msg
            print(f">>> created images and labels for {chipname}")

def normalize_image(image, **kwargs):
    """
    Normalize an image by rescaling its pixel values to a specified range.

    Args:
        image (ndarray): The input image to be normalized.
        **kwargs: Additional keyword arguments.
            percentiles (list, optional): The percentile range used to calculate the minimum and maximum values for normalization. Defaults to [1, 99].
            range (list, optional): The range to which the pixel values will be scaled. Defaults to [-1, 1].
            obsmask (ndarray, optional): An optional mask to apply to the image before normalization. Defaults to None.

    Returns:
        ndarray: The normalized image.

    """
    percentile_range = kwargs.get("percentiles", [1, 99])
    normal_scale_range = kwargs.get("srange", [-1, 1])
    obsmask = kwargs.get("obsmask", None)
    if obsmask is None:
        image_valid = image.astype("float")
    else:
        image_valid = image[np.where(obsmask)].astype("float")

    [min_val, max_val] = np.percentile(
        image_valid, percentile_range, interpolation="linear"
    )
    # normal_minmax_range = [min_val, max_val]
    # rescale
    image_scaled = ((image - min_val) / (max_val - min_val + C.EPS)) * (
        normal_scale_range[1] - normal_scale_range[0]
    ) + normal_scale_range[0]
    # bounding with the range
    image_scaled[np.where(image_scaled < normal_scale_range[0])] = normal_scale_range[0]
    image_scaled[np.where(image_scaled > normal_scale_range[1])] = normal_scale_range[1]
    return image_scaled

def normalize_datacube(datacube, **kwargs):
    """
    Normalize a datacube by rescaling each layer.

    Args:
        datacube (ndarray): The input datacube to be normalized.
        **kwargs: Additional keyword arguments.
            percentiles (list, optional): The percentiles used for rescaling. Default is [1, 99].
            srange (list, optional): The scale range used for rescaling. Default is [-1, 1].
            obsmask (ndarray, optional): The observation mask. Default is None.

    Returns:
        ndarray: The normalized datacube.

    """
    percentiles = kwargs.get("percentiles", [1, 99])
    srange = kwargs.get("srange", [-1, 1])
    obsmask = kwargs.get("obsmask", None)
    _datacube = datacube.copy() # not to alter the orginal values
    if C.MSG_FULL:
        print(f">>> normalizing the datacube to {srange} with percentiles {percentiles}")
    # rescale the data cube
    for i in range(0, _datacube.shape[0]):
        datalayer = normalize_image(_datacube[i, :, :], obsmask=obsmask, percentiles=percentiles, srange=srange)
        if obsmask is not None:
            datalayer[~obsmask] = min(srange) # give min value to the pixels out of the mask, as background
        _datacube[i, :, :] = datalayer
    return _datacube

def select_cloud_seed(
    cloud_mask, cloud_prob, label_cloud, label_noncloud, label_filled, seed_level_cloud, seed_level_noncloud
) -> np.ndarray:
    """Update the cloud mask for the cloud and non-cloud pixels with higher classification prob.

    Args:
        cloud_mask (2d array): cloud layer
        cloud_prob (2d array): cloud prob.
        label_cloud (number): pixel value for cloud
        label_noncloud (number): pixel value for noncloud
        label_filled (number): pixel value for filled
        seed_level_cloud (number): prctile of selecting cloud pixels
        seed_level_noncloud (number): prctile of selecting non-cloud pixels

    Returns:
        np.ndarray: mask of cloud, noncloud and filled
    """
    if seed_level_cloud == 0 & seed_level_noncloud == 0:  # all cloud and non-cloud masks
        return cloud_mask # directly return this seed mask because all pixels are included
        # will not reach here
        # hprob_cloud = np.min(cloud_prob[(cloud_mask == label_cloud)])
        # hprob_noncloud = np.min(cloud_prob[(cloud_mask == label_noncloud)])
    # selecting seeds if the seed level is not 0
    # for cloud seed
    _cloud_mask = cloud_mask.copy()
    if label_cloud in _cloud_mask:
        hprob_cloud = np.percentile(
            cloud_prob[(_cloud_mask == label_cloud)], q=seed_level_cloud
        )
        # exclude the pixels with lower classification prob.
        _cloud_mask[(cloud_prob < hprob_cloud) & (_cloud_mask == label_cloud)] = label_filled

    # for non-cloud seed
    if label_noncloud in _cloud_mask:
        hprob_noncloud = np.percentile(
            cloud_prob[(_cloud_mask == label_noncloud)], q=seed_level_noncloud
        )
        # exclude the pixels with lower classification prob.
        _cloud_mask[(cloud_prob < hprob_noncloud) & (_cloud_mask == label_noncloud)] = (
            label_filled
        )
    return _cloud_mask

def read_raster(des):
    """
    Read a raster dataset from a file. only singel band will be read.

    Args:
        des: The destination file path.

    Returns:
        data: The raster data.
        profile: The profile of the raster dataset.
    """
    with rasterio.open(des) as src:
        data = src.read(1)
        profile = src.profile
    return data, profile

def save_raster(data, profile, des, dtype=None):
    """
    Save a raster dataset to a file.

    Args:
        data: The raster data to be saved.
        profile: The profile of the raster dataset.
        des: The destination file path.

    Returns:
        None
    """
    # update the dtype accordingly
    if dtype is not None:
        profile["dtype"] = dtype
    else:
        profile["dtype"] = "uint8"
    # profile["driver"]="GTiff" # that make sure the rasterio does not compress the image. S2 data will be compressed by default with JPEG2000
    profile.update(driver='GTiff', compress='LZW', tiled=True) # losslss LZW compression 
    if os.path.isfile(des):
        os.remove(des)
    with rasterio.open(des, "w", **profile) as dst:
        dst.write(data, 1)

#%% Functions of showing images

def composite_rgb(red, green, blue, mask, percentiles=[2, 98], min_range = None):
    """
    Create a composite RGB image from individual red, green, and blue bands.

    Args:
        red (ndarray): Array representing the red band.
        green (ndarray): Array representing the green band.
        blue (ndarray): Array representing the blue band.
        mask (ndarray): Boolean array representing the mask.
        percentiles (list, optional): List of two percentiles used for contrast stretching. Defaults to [2, 98].
        min_range (float, optional): Minimum value for the range. Defaults to None. designed for cirrus band with mininum value 0.01

    Returns:
        ndarray: Composite RGB image.

    """
    _red = red.copy()
    _red[~mask] = np.nan
    _red_scale_range = np.nanpercentile(_red, percentiles)
    if min_range is not None:
        _red_scale_range[1] = max(_red_scale_range[1], min_range)
    _red = np.interp(_red, _red_scale_range, [0, 1])

    _green = green.copy()
    _green[~mask] = np.nan
    _green_scale_range = np.nanpercentile(_green, percentiles)
    if min_range is not None:
        _green_scale_range[1] = max(_green_scale_range[1], min_range)
    _green = np.interp(_green, _green_scale_range, [0, 1])

    _blue = blue.copy()
    _blue[~mask] = np.nan
    _blue_scale_range = np.nanpercentile(_blue, percentiles)
    if min_range is not None:
        _blue_scale_range[1] = max(_blue_scale_range[1], min_range)
    _blue = np.interp(_blue, _blue_scale_range, [0, 1])

    rgb = np.dstack([_red, _green, _blue])
    return rgb, _red_scale_range, _green_scale_range, _blue_scale_range

def show_image(rgb, title, path=None):
    """
    Display an RGB image.

    Parameters:
    rgb (numpy.ndarray): The RGB image to display.
    title (str): The title of the image.

    Returns:
    None
    """
    plt.imshow(rgb, interpolation="nearest")
    plt.axis("off")
    plt.title(title)
    if path is not None:
        # check it the path exists
        if not os.path.isfile(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0)
            plt.close()
    else:
        plt.show()

def show_simple_mask(mask, title):
    """
    Display a simple mask image with a single class.

    Args:
        mask (numpy.ndarray): The mask image.
        title (str): The title of the plot.

    Returns:
        None
    """
    plt.imshow(mask, cmap="gray", interpolation="none")
    plt.axis("off")
    plt.title(title)
    plt.show()

def show_cloud_mask(cloud_mask, classes, title):
    """
    Display the cloud mask image with color-coded classes.

    Args:
        cloud_mask (numpy.ndarray): The cloud mask image.
        classes (list): The list of class names.
        title (str): The title of the plot.

    Returns:
        None
    """

    # see color codes from https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
    label_dict = ["Filled", "Non-\ncloud", "Cloud"]
    color_dict = {1: "#cccccc", 2: "#000000", 3: "#fdae61"}
    cm = ListedColormap(color_dict.values())

    # convert the pixel value to the same as the index of label_dict
    cloud_mask_show = cloud_mask.copy()
    cloud_mask_show[cloud_mask == classes.index("filled")] = 1
    cloud_mask_show[cloud_mask == classes.index("noncloud")] = 2
    cloud_mask_show[cloud_mask == classes.index("cloud")] = 3

    # show the image
    im = plt.imshow(cloud_mask_show, cmap=cm, vmin=0.5, vmax=3.5, interpolation="none")
    plt.axis("off")
    plt.title(title)
    cbar = plt.colorbar(im, ticks=range(3))
    cbar.ax.get_yaxis().set_ticks([1, 2, 3])
    cbar.ax.get_yaxis().set_ticklabels(label_dict)
    plt.show()

def show_shadow_mask(shadow_mask, classes, title):
    """
    Display the shadow mask with different color codes for each class.

    Parameters:
    shadow_mask (numpy.ndarray): The shadow mask array.
    classes (list): The list of class names.
    title (str): The title of the plot.

    Returns:
    None
    """

    # see color codes from https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
    label_dict = ["Filled", "Non-\nshadow", "Shadow"]
    color_dict = {
        1: "#cccccc",
        2: "#000000",
        3: "#2c7bb6",
    }
    cm = ListedColormap(color_dict.values())

    # convert the pixel value to the same as the index of label_dict
    cloud_mask_show = shadow_mask.copy()
    cloud_mask_show[shadow_mask == classes.index("filled")] = 1
    cloud_mask_show[shadow_mask == classes.index("nonshadow")] = 2
    cloud_mask_show[shadow_mask == classes.index("shadow")] = 3

    # show the image
    im = plt.imshow(cloud_mask_show, cmap=cm, vmin=0.5, vmax=3.5, interpolation="none")
    plt.axis("off")
    plt.title(title)
    cbar = plt.colorbar(im, ticks=range(3))
    cbar.ax.get_yaxis().set_ticks([1, 2, 3])
    cbar.ax.get_yaxis().set_ticklabels(label_dict)
    plt.show()

def show_fmask_full(fmask, title):
    """
    Display the cloud mask image with color-coded classes, including land, water, and snow, shadow, cloud, and filled

    Args:
        cloud_mask (numpy.ndarray): The cloud mask image.
        title (str): The title of the plot.

    Returns:
        None
    """
    pass



def show_fmask(fmask, title, path=None, color_bar=False):
    """Display the cloud mask image with color-coded classes, including clear, shadow, cloud, and filled

    Args:
        cloud_mask (numpy.ndarray): The cloud mask image.
        title (str): The title of the plot.
        path (str): The path to save the plot. Defaults to None (will not save the figure).

    Returns:
        None
    """

    # see color codes from https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
    # https://a.atmos.washington.edu/~ovens/javascript/colorpicker.html
    label_dict = ["Filled", "Clear", "Shadow", "Cloud"]
    color_dict = {1: "#cccccc", 2: "#000000", 3: "#2c7bb6", 4: "#fdae61"}
    cm = ListedColormap(color_dict.values())

    # convert the pixel value to the same as the index of label_dict
    cloud_mask_show = fmask.copy()
    cloud_mask_show[fmask == C.LABEL_FILL] = 1
    cloud_mask_show[fmask == C.LABEL_LAND] = 2
    cloud_mask_show[fmask == C.LABEL_WATER] = 2
    cloud_mask_show[fmask == C.LABEL_SNOW] = 2
    cloud_mask_show[fmask == C.LABEL_SHADOW] = 3
    cloud_mask_show[fmask == C.LABEL_CLOUD] = 4

    # show the image

    im = plt.imshow(cloud_mask_show, cmap=cm, vmin=0.5, vmax=4.5, interpolation="none")
    plt.axis("off")
    plt.title(title)
    if color_bar:
        cbar = plt.colorbar(im, ticks=range(4))
        cbar.ax.get_yaxis().set_ticks([1, 2, 3, 4])
        cbar.ax.get_yaxis().set_ticklabels(label_dict)

    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=600, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

def show_seed_mask(seed_mask, cloud_mask, classes, title):
    """
    Display the seed mask with different colors representing different classes.

    Parameters:
    seed_mask (ndarray): The seed mask array.
    cloud_mask (ndarray): The cloud mask array.
    classes (list): The list of class labels.
    title (str): The title of the plot.

    Returns:
    None
    """

    # see color codes from https://colorbrewer2.org/#type=diverging&scheme=RdYlBu&n=5
    label_dict = ["Filled", "Non-\nCloud", "Cloud", "Non-\nCloud\nSeed", "Cloud\nSeed"]
    color_dict = {1: "#cccccc", 2: "#abd9e9", 3: "#fdae61", 4: "#2c7bb6", 5: "#d7191c"}
    cm = ListedColormap(color_dict.values())

    # convert the pixel value to the same as the index of label_dict
    cloud_mask_show = cloud_mask.copy()
    cloud_mask_show[cloud_mask == classes.index("filled")] = 1  # update filled
    cloud_mask_show[cloud_mask == classes.index("noncloud")] = 2  # update cloud mask
    cloud_mask_show[cloud_mask == classes.index("cloud")] = 3  # update non-cloud mask
    cloud_mask_show[seed_mask == classes.index("noncloud")] = 4  # update non-cloud seed
    cloud_mask_show[seed_mask == classes.index("cloud")] = 5  # update cloud seed

    # show the image
    im = plt.imshow(cloud_mask_show, cmap=cm, vmin=0.5, vmax=5.5, interpolation="none")
    plt.axis("off")
    plt.title(title)
    cbar = plt.colorbar(im, ticks=range(5))
    cbar.ax.get_yaxis().set_ticks([1, 2, 3, 4, 5])
    cbar.ax.get_yaxis().set_ticklabels(label_dict)
    plt.show()

def show_predictor(predictor, mask_filled, title, vrange = [0, 1]):
    """Display the predictor map with a colorbar.

    Args:
        cloud_prob (numpy.ndarray): The cloud probability map.
        mask_filled (numpy.ndarray): The mask for filled
        title (str): The title of the plot.

    Returns:
        None
    """
    if np.size(predictor) != np.size(mask_filled):
        pass # in case when Landsat 7 do not have the cirrus band and its prob.
        # raise ValueError("The size of the cloud probability and filled mask should be the same.")
    else:
        _cloud_prob = predictor.copy()
        _cloud_prob[mask_filled] = np.nan
        cm = mpl.colormaps.get_cmap("RdYlGn_r")
        cm.set_bad(color="#cccccc")  # gray for filled
        c = plt.imshow(_cloud_prob, vmin=vrange[0], vmax=vrange[1], cmap=cm, interpolation="nearest")
        plt.axis("off")
        plt.title(title)
        plt.colorbar(c)
        plt.show()

def show_cloud_probability(cloud_prob, mask_filled, title):
    """Display the cloud probability map with a colorbar.

    Args:
        cloud_prob (numpy.ndarray): The cloud probability map.
        mask_filled (numpy.ndarray): The mask for filled
        title (str): The title of the plot.

    Returns:
        None
    """
    if np.size(cloud_prob) != np.size(mask_filled):
        pass # in case when Landsat 7 do not have the cirrus band and its prob.
        # raise ValueError("The size of the cloud probability and filled mask should be the same.")
    else:
        _cloud_prob = cloud_prob.copy()
        _cloud_prob[mask_filled] = np.nan
        cm = mpl.colormaps.get_cmap("RdYlGn_r")
        cm.set_bad(color="#cccccc")  # gray for filled
        # plt.rcParams.update({'font.size': 22}) # using a larger font size
        plt.rcParams.update({'font.size': 14}) # using a larger font size
        c = plt.imshow(_cloud_prob, vmin=0, vmax=1, cmap=cm, interpolation="nearest")
        plt.axis("off")
        plt.title(title)
        plt.colorbar(c)
        plt.show()

def show_cloud_probability_hist(seed_cloud_prob, seed_noncloud_prob, prob_range, title = '', prob_bin=0.025):
    # calculate the density hist for each dataset with specified matching bin edges
    [prob_min, prob_max] = prob_range
    # in case when the prob_min is very close to prob_max, the bins will be empty
    prob_min = min(prob_min, 0)  # to make sure we have the full range of probablity
    prob_max = max(prob_max, 1)  # to make sure we have the full range of probablity
    bins_thrd = np.arange(prob_min, prob_max, prob_bin)
    bins_cloud, _ = np.histogram(seed_cloud_prob, bins=bins_thrd)
    bins_noncloud, _ = np.histogram(seed_noncloud_prob, bins=bins_thrd)
    # plot the histogram
    fig, ax = plt.subplots(figsize=(3, 2.5)) # setup the size of figure
    ax.bar(bins_thrd[:-1], bins_noncloud, width=prob_bin, color='#2c7bb6', align='edge', label='Non-cloud', alpha=0.8)
    ax.bar(bins_thrd[:-1], bins_cloud, width=prob_bin, color='#d7191c', align='edge', label='Cloud', alpha=0.8)
    ax.set_xlabel('Cloud Probability')
    ax.set_ylabel('Frequency')
    ax.legend()
    # Set the range of x-axis
    plt.xlim(0, 1) # on or off
    plt.title(title)
    
    from matplotlib.ticker import FormatStrFormatter
    from matplotlib.ticker import ScalarFormatter

    class ScalarFormatterClass(ScalarFormatter):
        def _set_format(self):
            self.format = "%1.2f"

    yScalarFormatter = ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0,0))
    ax.yaxis.set_major_formatter(yScalarFormatter)
    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2e')) ## Makes X-axis label with two decimal points
    plt.show()
        

def check_image_folder(directory, exclude=None):
    """check the image folder(s) in the directory

    Args:
        directory (string): Directory of the dataset, which can include landsat collection 2 level 1 or sentinel-2 level1c image folders
        exclude (list, optional): Images which is expected to excluded out of the progress if it is customized. Defaults to [].

    Returns:
        list: List of images founded in the directory (or list of the single imagery if it is)
    """

    # the name of folder
    folder_name = Path(directory).stem

    # Reference to https://docs.python.org/3/library/re.html

    # Level-1 of Landsat Collection 2 data format:
    # e.g., LC08_L1GT_001011_20140321_20200911_02_T2
    # (LT04|LT05|LE07|LC08|LC09) matches Landsat 4, 5, 7, 8, 9
    # L1[A-Za-z0-9]{2} matches Level 1 following two more alphanumeric characters, i.e., letters or digits
    # \d{6}_\d{8}_\d{8}_\d{2} macthes the path row, yyyymmdd for the acquisition, yyyymmdd for the product, and collection verstion
    # T\d{1} matches the Tile 1 or 2.
    # ^ and $ asserts position at the start and end of the string.
    # r indicates a raw string literal
    landsat_regex = (
        r"^(LT04|LT05|LE07|LC08|LC09)_L1[A-Za-z0-9]{2}_\d{6}_\d{8}_\d{8}_\d{2}_T\d{1}$"
    )

    # Level1C of Sentinel-2 data format:
    # e.g., S2A_MSIL1C_20160417T110652_N0201_R137_T29RPQ_20160417T111159.SAFE
    # S2[AB]_ matches "S2A_" or "S2B_" at the beginning of the identifier, representing Sentinel-2A or Sentinel-2B.
    # \d{8}T\d{6} matches a date and time in the format YYYYMMDDTHHMMSS
    # _\w{1,}_ matches one or more alphanumeric characters (letters or digits) separated by underscores
    sentinel_regex = (
        r"(S2A|S2B)_MSIL1C_\d{8}T\d{6}_\w{1,}_\w{1,}_\w{1,}_\d{8}T\d{6}\.SAFE$"
    )

    # if it is landsat collection 2 level 1
    if re.fullmatch(landsat_regex, folder_name):
        return [directory]
    # if Sentinel-2 level1c
    elif re.fullmatch(sentinel_regex, folder_name):
        return [directory]
    else:
        # search all images in the directory
        image_list = sorted(glob.glob(os.path.join(directory, "[L|S]*")))

        # filter out the images that disagree with the regex defined above
        image_list = [
            i for i in image_list if not re.fullmatch(landsat_regex, Path(i).stem)
        ]
        image_list = [
            i for i in image_list if not re.fullmatch(sentinel_regex, Path(i).stem)
        ]

        # exclude the images that do not contain cloud shadow layer
        if exclude is not None:
            image_list = [i for i in image_list if Path(i).stem not in exclude]
        return image_list

#%% Functions TBD
def excold_cloud(temperature=[], llow_temp=-999):
    if exist(temperature):
        mask_ccloud = temperature < llow_temp
        mask_cloud = np.logical_or(mask_cloud, mask_ccloud)
    else:
        mask_ccloud = []
    return mask_cloud, mask_ccloud

def examine_excold_cloud(mask_cloud, temperature=[], llow_temp=-999):
    if exist(temperature):
        mask_ccloud = temperature < llow_temp
        mask_cloud = np.logical_or(mask_cloud, mask_ccloud)
    else:
        mask_ccloud = []
    return mask_cloud, mask_ccloud

def calculate_slope(ele):
    """
    Calculate the slope of the elevation data.

    Args:
        ele (ndarray): The input elevation data.

    Returns:
        ndarray: The slope data.

    """
    # calculate the slope
    slope = np.arctan(np.sqrt(np.square(np.gradient(ele)[0]) + np.square(np.gradient(ele)[1])))
    return slope
    
# function enhance line
def enhance_line(band):
    """
    Enhances the input band by applying line enhancement filters.
    Parameters:
    - band: numpy.ndarray
        The input band to be enhanced.
    Returns:
    - numpy.ndarray
        The enhanced band after applying line enhancement filters.
    """
    
    # Template 1
    template = np.array([[-1, 2, -1],
                         [-1, 2, -1],
                         [-1, 2, -1]]) / 6.0
    line_enhanced = convolve(band, template)

    # Template 2
    template = np.array([[-1, -1, -1],
                         [ 2,  2,  2],
                         [-1, -1, -1]]) / 6.0
    line_enhanced_new = convolve(band, template)
    line_enhanced = np.maximum(line_enhanced, line_enhanced_new)

    # Template 3
    template = np.array([[ 2, -1, -1],
                         [-1,  2, -1],
                         [-1, -1,  2]]) / 6.0
    line_enhanced_new = convolve(band, template)
    line_enhanced = np.maximum(line_enhanced, line_enhanced_new)

    # Template 4
    template = np.array([[-1, -1,  2],
                         [-1,  2, -1],
                         [ 2, -1, -1]]) / 6.0
    line_enhanced_new = convolve(band, template)
    line_enhanced = np.maximum(line_enhanced, line_enhanced_new)

    return line_enhanced

# Function to fill NaN values row by row
def fill_nan_wise(arr):
    """
    Fills NaN values in a 2D array row-wise and column-wise.

    Args:
        arr (numpy.ndarray): The input 2D array.

    Returns:
        numpy.ndarray: The array with NaN values filled row-wise and column-wise.
    """
    arr = fill_nan_rowwise(arr)
    arr = fill_nan_colwise(arr)
    return arr

def fill_nan_rowwise(arr):
    for i in range(arr.shape[0]):
        row = arr[i, :]
        mask = np.isnan(row)
        if np.any(~mask):
            row[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), row[~mask])
    return arr

# Function to fill NaN values column by column
def fill_nan_colwise(arr):
    """
    Fill NaN values column-wise in a 2D array using linear interpolation.
    
    Parameters:
        arr (ndarray): The input 2D array.
        
    Returns:
        ndarray: The array with NaN values filled using linear interpolation.
    """
    for j in range(arr.shape[1]):
        col = arr[:, j]
        mask = np.isnan(col)
        if np.any(~mask):
            col[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), col[~mask])
    return arr

def exclude_images_by_tile(exclude, datasets = None, directory = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/ReferenceDataset"):
    """Exclude the images from the list of images

    Args:
        images (list): The list of images
        exclude (list): The list of tile to be excluded

    Returns:
        list: The list of tiles for excluding the images
    """
    # find all images that will be used as training samples
    if datasets is None:
        datasets = ["L8BIOME", "L8SPARCS", "L895CLOUD", "S2ALCD", "S2IRIS", "S2WHUCDPLUS", "S2FMASK"]
    images_excluded = []
    images_dataset = []
    for ds in datasets:
        path_image_list = sorted(glob.glob(os.path.join(directory, ds, "[L|S]*")))
        for img in path_image_list:
            img_name = Path(img).stem
            if img_name.startswith("L"):
                tile = img_name.split("_")[2]
            elif img_name.startswith("S"):
                tile = img_name.split("_")[5]
            if tile in exclude:
                images_excluded.append(img_name)
                if (ds not in images_dataset):
                    images_dataset.append(ds)
    return images_excluded, images_dataset


# End-of-file (EOF)
