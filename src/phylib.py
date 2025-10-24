"Physical rules to detect clouds"

import numpy as np
import copy
import pandas
import constant as C
from satellite import Data
import utils
from scipy.ndimage.filters import uniform_filter
from sklearn.linear_model import LinearRegression
from skimage.measure import label, regionprops


# np.seterr(invalid='ignore') # ignore the invalid errors

def mask_pcp(data: Data, satu):
    """mask possible cloud pixels (PCPs)

    Args:
        data (Data): datacube
        satu (2d array): saturation band

    Returns:
        bool: possible cloud pixels
    """
    # Basic test
    pcp = np.logical_and(
        np.logical_and(data.get("ndvi") < 0.8, data.get("ndsi") < 0.8),
        data.get("swir2") > 0.03,
    )

    # Temperature
    if data.exist("tirs1"):
        pcp = np.logical_and(pcp, data.get("tirs1") < 27)  # in degree

    # Whiteness test
    pcp = np.logical_and(pcp, data.get("whiteness") < 0.7)

    # Haze test
    pcp = np.logical_and(pcp, np.logical_or(data.get("hot") > 0, satu))

    # Ratio 4/5 test
    pcp = np.logical_and(
        pcp,
        (data.get("nir") / (data.get("swir1") + C.EPS)) > 0.75,
    )

    return pcp


def mask_snow(data: Data):
    """It takes every snow pixels including snow pixel under thin clouds or icy clouds

    Args:
        data (Data): datacube

    Returns:
        bool: snow/ice pixels
    """
    snow = np.logical_and(
        np.logical_and(
            data.get("ndsi") > 0.15,
            data.get("nir") > 0.11,
        ),
        data.get("green") > 0.1,
    )
    if data.exist("tirs1"):
        snow = np.logical_and(snow, data.get("tirs1") < 10)  # in degree
    return snow


def mask_abs_snow(data: Data, green_satu, snow, radius=167):
    """Select absolute snow/ice pixels using spectral-contextual for polar regions where large area of snow/ice (see Qiu et al., 2019)"

    Args:
        data (Data): datacube
        green_satu (bool): Saturation of the green band
        snow (bool): spectral-based snow/ice mask
        radius (int, optional): Kernel size in pixels. Defaults to 167.

    Returns:
        bool: snow/ice pixels
    """

    # radius = 2*radius + 1 # as to the window size which is used directly
    # green_var = uniform_filter((data.get("green")*data.get("green")).astype(np.float32), radius, mode='reflect') - np.square(uniform_filter((data.get("green")).astype(np.float32), radius, mode='reflect')) # must convert to float32 or 64 to get the uniform_filter
    # green_var[green_var<0] = C.EPS # Equal to 0
    # absnow = np.logical_and(np.logical_and(np.sqrt(green_var)*np.sqrt(radius*radius/(radius*radius-1))*(1-data.get("ndsi")) < 0.0009, snow), ~green_satu) # np.sqrt(green_var)*(1-ndsi) < 9 is SCSI
    # Note np.sqrt(radius*radius/(radius*radius-1)) is to convert it as same as that of matlab function, stdflit, see https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html

    # only get snow/ice pixels from all potential snow/ice pixels, and
    # do not select the saturated pixels at green band which may be cloud!
    # Local standard deviation of image's ref: https://stackoverflow.com/questions/18419871/improving-code-efficiency-standard-deviation-on-sliding-windows
    # and https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    radius = 2 * radius + 1  # as to the window size which is used directly
    green_var = uniform_filter(
        (data.get("green").astype(np.float32)) ** 2,
        radius,
        mode="reflect",
    ) - np.square(
        uniform_filter(
            data.get("green").astype(np.float32),
            radius,
            mode="reflect",
        )
    )  # must convert to float32 or 64 to get the uniform_filter .astype(np.float32)
    green_var[green_var < 0] = C.EPS  # Equal to 0
    absnow = np.logical_and(
        np.logical_and(
            np.sqrt(green_var)
            * np.sqrt(radius**2 / (radius**2 - 1))
            * (1 - data.get("ndsi"))
            < 0.0009,
            snow,
        ),
        ~green_satu,
    )  # np.sqrt(green_var)*(1-ndsi) < 9 is SCSI
    # Note np.sqrt(radius**2/(radius**2-1)) is to convert it as same as that of matlab function, stdflit, see https://nickc1.github.io/python,/matlab/2016/05/17/Standard-Deviation-(Filters)-in-Matlab-and-Python.html
    return absnow


def split_to_zones(min_dem, max_dem, step_dem):
    """Splite the range from min to max into multiple zones, with a step of size. This version is some different from the Matlab version

    Args:
        min_dem (float): Mininum DEM
        max_dem (float): Maximum DEM
        step_dem (float): Step

    Returns:
        list of zones: _description_
    """
    splits = np.arange(min_dem, max_dem, step_dem)
    # in case we do not split the dem
    if len(splits) == 0:
        return None
    # regular processing
    zones = []
    for i, start in enumerate(splits):  # every 100 meters
        if i == 0:
            zone = [float("-inf"), start + step_dem]  # first zone
        elif i == len(splits) - 1:
            zone = [start, float("inf")]  # last zone
        else:
            zone = [start, start + step_dem]  # other zones
        zones.append(zone)
    return zones


def normalize_cirrus(cirrus, clear, obsmask=None, dem=None, dem_min=None, dem_max=None):
    """Normlize cirrus band based on DEM

    Args:
        cirrus (float): Cirrus band in reflectance, that will be varied
        clear (bool): Clear pixels in the image
        obsmask (bool, optional): Observation mask. Defaults to None.
        dem (float, optional): DEM data. Defaults to None.
        dem_min (float, optional): Mininum DEM in the image. Defaults to None.
        dem_max (float, optional): Maxinum DEM in the image. Defaults to None.

    Returns:
        float: Normalized cirrus band
    """

    if np.any(
        np.logical_and(clear, obsmask)
    ):  # if no abs clear pixels, just return the original cirrus band
        _cirrus = cirrus.copy()  # do not alter the original data
        # clear = np.logical_and(clear, obsmask) # updated with the mask at the first
        if dem is not None:
            if dem_min < dem_max:  # all the pixels are not the same, in case zeros
                dem_zones = split_to_zones(dem_min, dem_max, 100)
                clear_base = 0  # clear pixels's based cirrus band reflectance
                for zone in dem_zones:
                    dem_zone = np.logical_and(
                        obsmask, np.logical_and(dem > zone[0], dem <= zone[1])
                    )
                    if dem_zone.any():
                        clear_zone = np.logical_and(clear, dem_zone)
                        if clear_zone.any():
                            clear_base = get_percentile(_cirrus, clear_zone, 2)
                            # 2 percentile as dark pixel # updated as the newest base
                        _cirrus[dem_zone] = (
                            _cirrus[dem_zone] - clear_base
                        )  # in the case when we do not get the base, we can fetch up the previous one as base to continue to
            else:
                _cirrus = _cirrus - get_percentile(
                    _cirrus, np.logical_and(clear, obsmask), 2
                )
        else:  # cirrus band is adjusted to 0-level even when there is no DEM
            _cirrus = _cirrus - get_percentile(
                _cirrus, np.logical_and(clear, obsmask), 2
            )
            # 2 percentile as dark pixel
        _cirrus[_cirrus <= 0] = (
            0  # the normalized cirrus value will be set as 0 when it is negative
        )
        return _cirrus
    else:
        return cirrus


def normalize_temperature(temperature, dem, dem_min, dem_max, clear_land, number):
    """Normalize the thermal band

    Args:
        temperature (float): Thermal band in degree
        dem (float): DEM in meter
        dem_min (float, optional): Mininum DEM in the image. Defaults to None.
        dem_max (float, optional): Maxinum DEM in the image. Defaults to None.
        clear_land (float): Clear land pixels
        number (int): Number of pixels for fitting model

    Returns:
        float: Normalized thermal
    """
    _temperature = temperature.copy()  # do not alter the original data
    # clear_land = np.logical_and(clear_land, obsmask) # updated with the mask at the first
    # only use the high confident levels to estimate
    [temp_low, temp_high] = np.percentile(
        _temperature[clear_land], [C.LOW_LEVEL, C.HIGH_LEVEL]
    )
    samples = stratify_samples(
        np.logical_and(
            clear_land,
            np.logical_and(_temperature >= temp_low, _temperature <= temp_high),
        ),  # do not wnat change its original value
        dem,
        dem_min,  # np.percentile(dem[obsmask], 0.001)
        dem_max,  # 0.001 for excluding the noises of DEM data
        number=number,  # total number of sample pixels we expected
        step=300,
        distance=15,
    )  # mininum distance: unit: pixels
    if (
        len(samples) > 20
    ):  # 10 by 2 (parameters, i.e., slope and intercept); A common rule of thumb is to have at least 10 times as many samples as there are parameters to be estimated.
        # Estmate the lapse_rate of temperature by Ordinary least squares Linear Regression.
        reg = LinearRegression().fit(
            dem[samples.row, samples.column].reshape(-1, 1),
            _temperature[samples.row, samples.column],
        )
        # reg.coef_[0] is the slope, which is treated as the rate, that must be negative pysically
        if (
            reg.coef_[0] < 0
        ):  # only when the rate is negative pysically, we normalize the temperature band
            _temperature = _temperature - reg.coef_[0] * (
                dem - dem_min
            )  # the pixels located at higher elevation will be normalized to be one with higher temperature
    return _temperature, temp_low, temp_high


def stratify_samples(clear, data, data_min, data_max, step, number, distance=15):
    """stratified sampling DEM

    Args:
        clear (bool): Clear pixels
        data (2d array): Data inputted
        data_min (float): Mininum data
        data_max (float): Maxinum data
        step (float): Step interval to split data
        number (int, optional): Number of pixels seleceted.
        distance (int, optional): Mininum distance among the sampling pixels in pixels. Defaults to 15.

    Returns:
        dataframe: sampling pixels in row and column
    """

    data_zones = split_to_zones(data_min, data_max, step)

    df_sample_selected = []
    if (
        data_zones is not None
    ):  # this was because sometimes min value == max value in DEM
        number = round(number / len(data_zones))
        # equal samples in each stratum

        # we create a basic layer for sampling, we locatd a pixel every min_distance, and later on we only picked up the sample overlapped this base layer. This will be faster than previous MATLAB version, but similar performance
        if distance > 0:
            sampling_dist_base_layer = np.zeros(
                data.shape, dtype=bool
            )  # create a boolean base layer
            sampling_dist_base_layer[
                np.ix_(
                    range(0, sampling_dist_base_layer.shape[0], distance),
                    range(0, sampling_dist_base_layer.shape[1], distance),
                )
            ] = True  # _ix can quickly construct index arrays that will index the cross product.

        # at each isolate zone, we go to fetch the samples
        for zone in data_zones:
            clear_zone = np.logical_and.reduce(
                [clear, zone[0] < data, data <= zone[1]]
            )  # at the isolated zone
            if clear_zone.any():  # if any pixels located
                if distance > 0:  # over the mininum distrance base layer
                    clear_zone = np.logical_and(clear_zone, sampling_dist_base_layer)
                df_clear_zone = pandas.DataFrame(
                    np.argwhere(
                        clear_zone
                    ),  # pick up the pixels located in the isolated zone
                    columns=["row", "column"],
                )  # creating df object with columns specified
                del clear_zone
                df_clear_zone = df_clear_zone.sample(
                    np.min(
                        [number, len(df_clear_zone.index)]
                    ),  # when the number of data is smaller than that we expected
                    random_state=C.RANDOM_SEED,  # static seed for random
                    ignore_index=True,
                )  # unnecessery to update the index

                # append to the final
                df_sample_selected.append(df_clear_zone)
                del df_clear_zone
        if len(df_sample_selected) > 0:
            df_sample_selected = pandas.concat(df_sample_selected)
    return df_sample_selected


def mask_water(data: Data, obsmask, snow, swo_erosion_radius = 0):
    """the spectral-based water mask (works over thin cloud)

    Args:
        data (Data): datacube
        obsmask (bool): Observation mask of the image
        snow (float): snow/ice mask

    Returns:
        bool: water pixels
    """
    water = np.logical_or(
        np.logical_and(
            np.logical_and(
                data.get("ndvi") > 0,
                data.get("ndvi") < 0.1,
            ),
            data.get("nir") < 0.05,
        ),
        np.logical_and(
            data.get("ndvi") < 0.01,
            data.get("nir") < 0.11,
        ),
    )
    water = np.logical_and(water, obsmask)
    # the swo-based water mask
    if water.any() and data.exist(
        "swo"
    ):  # when water pixels were identifed and the swo is available
        swo = data.get("swo")  # to get the layer
        if swo_erosion_radius > 0:
            # erosion the swo layer to exclude narrow water bodies, like rivers. in this case, we do not reply on the swo layer to mask water pixels, but the spectral rules still work on these pixels
            swo[~utils.erode(swo>0, swo_erosion_radius)]= 0 # note this will change the orginal layer!
        # low level (17.5%) to exclude the commssion errors as water.
        # 5% tolerances.
        swo_thrd = np.percentile(swo[water], C.LOW_LEVEL, method="midpoint") - 5
        # merge the spectral-based water mask and the swo-based water mask
        if swo_thrd > 0:
            water = np.logical_or(
                water, np.logical_and(swo > swo_thrd, ~snow)
            )  # get the swo-based water mask, exclude snow/ice over ocean
    return water


def probability_cirrus(cirrus):
    """Cloud probability of cirrus

    Args:
        cirrus (float): Cirrus band reflectance

    Returns:
        float: Cloud probability of cirrus
    """
    prob_cir = np.clip(cirrus / 0.04, 0, None)
    return prob_cir


def probability_land_varibility(data: Data, green_satu, red_satu):
    """Cloud probability of varibility for land

    Args:
        data (Data): datacube
        green_satu: bool: Saturation of the green band
        red_satu: bool: Saturation of the red band

    Returns:
        float: Cloud probability of varibility for land
    """
    # prob_land_var = 1 - np.amax(
    #     [np.abs(ndvi), np.abs(ndsi), np.abs(ndbi), whiteness], axis=0
    # )
    # fixed the saturated visible bands for NDVI and NDSI
    _ndvi = np.where(
        np.bitwise_and(green_satu, data.get("ndvi") < 0), 0, data.get("ndvi")
    )
    _ndsi = np.where(
        np.bitwise_and(red_satu, data.get("ndsi") < 0), 0, data.get("ndsi")
    )
    return 1 - np.maximum(
        np.maximum(
            np.maximum(
                np.abs(_ndvi),
                np.abs(_ndsi),
            ),
            np.abs(data.get("ndbi")),
        ),
        data.get("whiteness"),
    )


def probability_land_temperature(temperature, clear_land):
    """Calculate the probability of land temperature based on the given temperature array and clear land mask.

    Parameters:
    temperature (ndarray): Array of temperature values.
    clear_land (ndarray): Boolean mask indicating clear land areas.

    Returns:
    prob_land_temp (ndarray): Array of probabilities of land temperature.
    temp_low (float): Lower threshold temperature.
    temp_high (float): Upper threshold temperature.
    """

    [temp_low, temp_high] = np.percentile(
        temperature[clear_land], [C.LOW_LEVEL, C.HIGH_LEVEL]
    )
    temp_low = temp_low - 4  # 4 C-degrees
    temp_high = temp_high + 4  # 4 C-degrees
    prob_land_temp = (temp_high - temperature) / (temp_high - temp_low)
    prob_land_temp = np.clip(prob_land_temp, 0, None)
    return prob_land_temp


def probability_water_temperature(temperature, clear_water):
    """Calculate the probability of water temperature based on the given temperature array and clear water mask.

    Parameters:
    temperature (numpy.ndarray): Array of temperature values.
    clear_water (numpy.ndarray): Boolean mask indicating clear water pixels.

    Returns:
    numpy.ndarray: Array of probabilities of water temperature.

    """
    prob_water_temp = (
        np.percentile(temperature[clear_water], C.HIGH_LEVEL) - temperature
    ) / 4  # 4 degree to normalize
    prob_water_temp = np.clip(prob_water_temp, 0, None)
    return prob_water_temp


def probability_water_brightness(data: Data):
    """
    Calculate the probability of water brightness for each pixel in the input data.

    Parameters:
    data (Data): The input data containing the necessary bands.

    Returns:
    prob_water_bright (numpy.ndarray): The probability of water brightness for each pixel.
    """
    prob_water_bright = data.get("swir1") / 0.11
    prob_water_bright = np.clip(prob_water_bright, 0, 1)
    return prob_water_bright


def probability_land_brightness(data, clear_land):
    """
    Calculate the probability of land brightness for a given set of hot and clear land values.

    Parameters:
    data (Data): The input data containing the necessary bands.
    clear_land (numpy.ndarray): Array of clear land values.

    Returns:
    numpy.ndarray: Array of probabilities of land brightness.

    """
    [hot_low, hot_high] = np.percentile(
        data.get("hot")[clear_land], [C.LOW_LEVEL, C.HIGH_LEVEL]
    )
    hot_low = hot_low - 0.04  # 0.04 reflectance
    hot_high = hot_high + 0.04  # 0.04 reflectance
    prob_land_bright = (data.get("hot") - hot_low) / (hot_high - hot_low)
    prob_land_bright = np.clip(
        prob_land_bright, 0, 1
    )  # 1  # this cannot be higher 1 (maybe have commission errors from bright surfaces).
    return prob_land_bright


def flood_fill_shadow(nir_full, swir1_full, abs_land, obsmask,
                      threshold=0.15, nir_background=None, swir1_background=None):
    """
    Masks potential shadow areas in the input images based on flood fill method.

    Parameters:
        nir_full (numpy.ndarray): Array representing the NIR band image.
        swir1_full (numpy.ndarray): Array representing the SWIR 1 band image.
        abs_land (numpy.ndarray): Array representing the land mask.
        obsmask (numpy.ndarray): Array representing the observation mask.
        thershold (float, optional): The threshold value for the shadow mask. Defaults to 0.02.

    Returns:
        numpy.ndarray: Array representing the mask of potential shadow areas.
    """

    # mask potential shadow using flood fill method in NIR and SWIR 1 band
    # making surface data >=0 for the array
    # nir_full = np.maximum(0, nir_full)  # avoiding negative values
    # swir1_full = np.maximum(0, swir1_full)  # avoiding negative values
    # add "and (np.any(abs_land)" to avoid the error when there is no land pixels
    if (nir_background is None) and (np.any(abs_land)):
        nir_background = np.percentile(nir_full[abs_land], C.LOW_LEVEL)
    else:
        nir_background = nir_full[obsmask].min() # avoiding negative values
    if (swir1_background is None) and (np.any(abs_land)):
        swir1_background = np.percentile(swir1_full[abs_land], C.LOW_LEVEL)
    else:
        swir1_background = swir1_full[obsmask].min() # avoiding negative values
    
    nir_filled = utils.imfill(nir_full, obsmask, fill_value=nir_background)
    swir1_filled = utils.imfill(swir1_full, obsmask, fill_value=swir1_background)

    return (
        np.minimum((nir_filled - nir_full)/nir_filled,  (swir1_filled - swir1_full)/swir1_filled) > threshold
    )

def flood_fill_shadow2(nir_full, swir1_full, abs_land, obsmask, threshold=0.02, nir_background=None, swir1_background=None):
    """
    Masks potential shadow areas in the input images based on flood fill method.

    Parameters:
        nir_full (numpy.ndarray): Array representing the NIR band image.
        swir1_full (numpy.ndarray): Array representing the SWIR 1 band image.
        abs_land (numpy.ndarray): Array representing the land mask.
        obsmask (numpy.ndarray): Array representing the observation mask.
        thershold (float, optional): The threshold value for the shadow mask. Defaults to 0.02.

    Returns:
        numpy.ndarray: Array representing the mask of potential shadow areas.
    """

    # mask potential shadow using flood fill method in NIR and SWIR 1 band
    # making surface data >=0 for the array
    # nir_full = np.maximum(0, nir_full)  # avoiding negative values
    # swir1_full = np.maximum(0, swir1_full)  # avoiding negative values
    if nir_background is None:
        nir_background = np.percentile(nir_full[abs_land], C.LOW_LEVEL)
    else:
        nir_background = nir_full[obsmask].min() # avoiding negative values
    if swir1_background is None:
        swir1_background = np.percentile(swir1_full[abs_land], C.LOW_LEVEL)
    else:
        swir1_background = swir1_full[obsmask].min() # avoiding negative values
        
    return (
        np.minimum(
            utils.imfill(nir_full, obsmask, fill_value=nir_background)
            - nir_full,
            utils.imfill(swir1_full, obsmask, fill_value=swir1_background)
            - swir1_full,
        )
        > threshold
    )


def get_percentile(data, obsmask, pct):
    """get percentile value

    Args:
        data (number): data layer
        obsmask (bool): observation mask
        pct (number): percentile value

    Returns:
        number: _description_
    """
    return np.percentile(data[obsmask], pct)


def compute_cloud_probability_layers(image, min_clear, swo_erosion_radius=0, water_erosion_radius = 0):
    """Compute cloud probability layers according to the datacube

    Args:
        image (Object): Landsat or Sentinel-2 object
        saturation (2d array): Saturation mask
        min_clear (number, optional): Mininum number for further analyse.

    Returns:
        various varibles: all components regarding cloud probabilities
    """

    # shared variables (between functions) with start of the dash _
    _dem_min, _dem_max = get_percentile(
        image.data.get("dem"), image.obsmask, [0.001, 99.999]
    )

    # _dem_max = get_percentile(data[bands.index("dem"), :, :], image.obsmask, 99.999)

    ## identify Potential Cloud Pixels (PCPs)
    pcp = mask_pcp(image.data, image.get_saturation(band="visible"))
    snow = mask_snow(image.data)

    # Exclude absolute snow pixels out of PCPs
    if np.count_nonzero(snow) >= np.square(
        10000 / image.resolution
    ):  # when the snow pixels are enough
        # exclude absolute snow/ice pixels
        pcp[
            mask_abs_snow(
                image.data,
                image.get_saturation(band="green"),
                snow,
                radius=np.ceil(5000 / image.resolution),
            )
        ] = False

    # Appen thin cloud pixels to PCPs
    if image.data.exist("cirrus"):
        cirrus = normalize_cirrus(
            image.data.get("cirrus"),
            ~pcp,
            obsmask=image.obsmask,
            dem=image.data.get("dem"),
            dem_min=_dem_min,
            dem_max=_dem_max,
        )
        pcp = np.logical_or(
            pcp, cirrus > 0.01
        )  # Update the PCPs with cirrus band TOA > 0.01, which may be cloudy as well

    # ABS CLEAR PIXELs with the observation extent
    # This can save the 'not' operation at next processings
    abs_clr = np.logical_and(
        ~pcp, image.obsmask
    )  # convert pcp as clear and updated it with the obs. mask

    # Seperate absolute clear mask into land and waster groups
    # mask water no mater if we go further to analyze the prob.
    if swo_erosion_radius < 0: # should be in meters, if the swo_erosion_radius is negative, it means the radius is in meters
        water = mask_water(image.data, image.obsmask, snow=snow, swo_erosion_radius = int(np.abs(np.ceil(swo_erosion_radius / image.resolution))))
    else: # should be in pixels, if the swo_erosion_radius is positive, it means the radius is in pixels
        water = mask_water(image.data, image.obsmask, snow=snow, swo_erosion_radius = swo_erosion_radius)

    # that will be used if the thermal band is available
    surface_low_temp, surface_high_temp = None, None

    # Start to anaylze cloud prob.
    if np.count_nonzero(abs_clr) <= min_clear:
        # Case 1: special case when there are lots of cloudy pixels
        # mask_cloud  = ~mask_pcp # all PCPs were masked as cloud directly because of no enought clear pixels for further analyses
        # mask_water = utils.init_mask(IMAGE.obsmask, dtype = 'bool', defaultvalue = 0) # no water pixels, in order to merge all the layers into 1 layer at the end
        # mask potential shadow
        activated = False
        # pcp, done above
        lprob_var = None
        lprob_temp = None
        wprob_temp = None
        wprob_bright = None
        prob_cirrus = None
        # water, done above
        # snow, done above
    else:
        # Case 2: regular cases, with cloud prob.

        # Fist of all, to check out cirrus band thermal -based probs. and once they are conducted, we can empty the data
        # Cloud probability: thin cloud (or cirrus) probability for both water and land
        if image.data.exist("cirrus"):
            prob_cirrus = probability_cirrus(cirrus)
            del cirrus  # that can be deleted
        else:
            prob_cirrus = 0
        
        # erode the water pixels to exclude the narrow water bodies, like rivers and coastal lines
        if water_erosion_radius < 0: # should be in meters, if the swo_erosion_radius is negative, it means the radius is in meters
            water = utils.erode(water, int(np.abs(np.ceil(swo_erosion_radius / image.resolution))))
        elif water_erosion_radius > 0: # should be in pixels, if the swo_erosion_radius is positive, it means the radius is in pixels
            water = utils.erode(water, water_erosion_radius)

        abs_land = np.logical_and(~water, abs_clr)
        # Check the number of absolute clear pixels, and when not enough, fmask goes back to pick up all
        if np.count_nonzero(abs_land) <= min_clear:
            abs_land = abs_clr
        abs_water = np.logical_and(water, abs_clr)
        del abs_clr

        # OVER LAND #
        # Cloud probability: varibility probability over land
        lprob_var = probability_land_varibility(
            image.data,
            image.get_saturation(band="green"),
            image.get_saturation(band="red"),
        )

        # Cloud probability: temperature and brightness probability over land
        if image.data.exist("tirs1"):
            tirs1, surface_low_temp, surface_high_temp = normalize_temperature(
                image.data.get("tirs1"),
                image.data.get("dem"),
                _dem_min,
                _dem_max,
                abs_land,
                min_clear,
            )
            lprob_temp = probability_land_temperature(
                tirs1, abs_land
            )  # that has been normalized
        else:
            lprob_temp = probability_land_brightness(image.data, abs_land)

        # END of LAND #

        # OVER WATER #
        if np.any(
            abs_water
        ):  # Only when the clear water pixels are identified, this will be triggered.
            # Cloud probability: temperature probability over water
            if image.data.exist("tirs1"):
                wprob_temp = probability_water_temperature(tirs1, abs_water)
                del tirs1
            else:
                wprob_temp = 1.0
            # Cloud probability: brightness probability over water
            wprob_bright = probability_water_brightness(image.data)
        else:
            wprob_temp = None
            wprob_bright = None
        # END of WATER #

        # empty the saturation data, which will not be used any more
        image.clean_saturation()
        
        activated = True
    return (
        activated,
        pcp,
        lprob_var,
        lprob_temp,
        wprob_temp,
        wprob_bright,
        prob_cirrus,
        water,
        snow,
        surface_low_temp,
        surface_high_temp,
    )


def combine_cloud_probability(
    var, tmp, cir, prob_var, prob_temp, prob_cirrus, woc, mask_absclear, adjusted=True
):
    """combine cloud probability layer according to the physical rules

    Args:
        var (bool): True to control the variation probability
        tmp (bool): True to control the temporal probability
        cir (bool): True to control the cirrus probability
        prob_var (2d array): variation probability
        prob_temp (2d array): temporal probability
        prob_cirrus (2d array): cirrus probability
        woc (number): Weight of cirrus probability
        mask_absclear (2d array in bool): Clear pixels
        adjust (bool, optional): Adjust the cloud probability. Defaults to True.

    Returns:
        2d array: Combined cloud probability
    """
    # combine cloud probabilities for land or water
    if var and tmp:
        prob = prob_var * prob_temp
    elif var:
        # copy the prob_var to prob, and then update the prob according to the mask_absclear
        prob = prob_var
    elif tmp:
        prob = prob_temp
    else:
        prob = 0
    if cir:
        prob = prob + woc * prob_cirrus
    if adjusted:
        prob = prob - clear_probability(prob, mask_absclear)
        prob[prob < 0] = 0 # we will set it as 0 in final prob.
    # prob[mask_absclear] = 0 # no need. exclude abs clear pixels from the cloud probability as 0
    return prob


def convert2seedgroups(
    mask_prob, seed, label_cloud, label_noncloud, bin_width=0.025, equal_num=False
):
    """
    Convert the mask probabilities of seed pixels into two groups: cloud and non-cloud.

    Args:
        mask_prob (numpy.ndarray): Array of mask probabilities.
        seed (numpy.ndarray): Array of seed labels.
        label_cloud: Label for cloud pixels.
        label_noncloud: Label for non-cloud pixels.
        bin_width (float, optional): Width of the probability bins. Defaults to 0.025.
        equal_num (bool, optional): Whether to have an equal number of seed pixels between cloud and non-cloud.
                                   Defaults to True.

    Returns:
        tuple: A tuple containing:
            - seed_cloud_prob (numpy.ndarray): Array of mask probabilities for cloud seed pixels.
            - seed_noncloud_prob (numpy.ndarray): Array of mask probabilities for non-cloud seed pixels.
            - prob_range (list): Range of mask probabilities.

    """
    seed_cloud_prob = mask_prob[seed == label_cloud].flatten()
    seed_noncloud_prob = mask_prob[seed == label_noncloud].flatten()

    # 0.05% was used to exclude potential anormly data
    # merge the two groups of seed pixels
    prob_range = np.percentile(
        np.concatenate([seed_cloud_prob, seed_noncloud_prob]),
        q=[0.05, 99.95],
    )
    prob_range = [
        np.floor(prob_range[0] / bin_width) * bin_width,
        np.ceil(prob_range[1] / bin_width) * bin_width,
    ]  # make the range into 0.025 level

    # in case when the prob_min is very close to prob_max, the bins will be empty
    prob_range[0] = min(
        prob_range[0], 0
    )  # to make sure we have the full range of probability
    prob_range[1] = max(
        prob_range[1], 1
    )  # to make sure we have the full range of probability

    # adjust the values to avoid the discarded pixels in final total number
    seed_cloud_prob[seed_cloud_prob < prob_range[0]] = prob_range[0]
    seed_cloud_prob[seed_cloud_prob > prob_range[1]] = prob_range[1]

    if equal_num:
        np.random.seed(C.RANDOM_SEED)
        # same number of seed pixels between cloud and non-cloud
        if seed_cloud_prob.size > seed_noncloud_prob.size:
            seed_cloud_prob = seed_cloud_prob[
                np.random.choice(
                    seed_cloud_prob.size, seed_noncloud_prob.size, replace=False
                )
            ]
        elif seed_cloud_prob.size < seed_noncloud_prob.size:
            seed_noncloud_prob = seed_noncloud_prob[
                np.random.choice(
                    seed_noncloud_prob.size, seed_cloud_prob.size, replace=False
                )
            ]

    return seed_cloud_prob, seed_noncloud_prob, prob_range


def overlap_cloud_probability(
    seed_cloud_prob,
    seed_noncloud_prob,
    prob_range=None,
    prob_bin=0.025,
    threshold=0,
    split=True,
):
    """find the overlapping density of cloud and non-cloud pixels and the optimal thershold to separate them

    Args:
        mask_prob (2d array): physical-based cloud probability
        mask_seed (2d array): mask of cloud and non-cloud layer, in which the cloud is 1 and the non-cloud is 0, and filled pixels are provided with a different value
        label_cloud (int, optional): pixel value indicating cloud. Defaults to 1.
        label_noncloud (int, optional): pixel value indicating noncloud. Defaults to 0.
        prob_range (list, optional): range of cloud probability. Defaults to [0, 1].
        prob_bin (float, optional): width of bins of cloud probability. Defaults to 0.025.
        threshold (float, optional): threshold to separate cloud and non-cloud pixels. Defaults to 0.05.
        split (bool, optional): split cloud and non-cloud pixels by a thershold. Defaults to True.

    Returns:
        number, number: overlapping density and optimal thershold
    """

    # calculate the density hist for each dataset with specified matching bin edges
    if prob_range is None:
        prob_range = [0, 1]
    [prob_min, prob_max] = prob_range

    if prob_max > prob_min:
        bins_thrd = np.arange(
            prob_min, prob_max + prob_bin, prob_bin
        )  # + prob_bin to make sure the prob_max is included
    else:  # in case when the prob_min is same as to prob_max, the bins will be empty
        bins_thrd = [prob_min, prob_min + prob_bin]

    bins_cloud, _ = np.histogram(seed_cloud_prob, bins=bins_thrd)
    bins_noncloud, _ = np.histogram(seed_noncloud_prob, bins=bins_thrd)
    # e.g., if bans are [0, 1, 2, 3], the counts will be [0, 1) [1, 2) [2, 3)
    # total_pixels = len(seed_cloud_prob) + len(seed_noncloud_prob)

    # calculate the overlapping density
    # bins_overlap = np.min([bins_cloud, bins_noncloud], axis=0)
    over_rate = (np.min([bins_cloud, bins_noncloud], axis=0)).sum() / (
        len(seed_cloud_prob) + len(seed_noncloud_prob)
    )

    if split:  # to get the optimal thershold
        # search optimal thershold to seperate cloud and non-cloud pixels
        # as we do not want to miss cloud pixels in the physical layer, we will set the threshold_buffer as 0.95
        thrd_record = bins_thrd[
            -1
        ]  # default with maximum error rate by setting the thershold as 1.0
        num_errors_record = (
            bins_cloud.sum()
        )  # in this case, all cloudy pixels were misclassified as non-cloud
        bins_thrd = bins_thrd[:-1]  # remove the last bin's right boundary
        for i, thrd in enumerate(bins_thrd):
            # max_overlap = np.max([bins_overlap[0:i].sum(), bins_overlap[i:].sum()])
            # count # of cloud pixels on left of the x-axis and # of non-cloud pixels on right of the x-axis, and if their sum moves to be smaller, the result is optimal
            # the thershold of segmenting clouds is "> thrd" rather than ">= thrd", because the thershold is the left boundary of the bin (included), but the right boundary is not included.
            # cloud seed pixels were counted by "< thrd" and non-cloud seed pixels were counted by ">= thrd"
            num_errors = (
                bins_cloud[
                    0:i
                ].sum()  # cloud pixels on the left of the thershold, not included the boundary
                + bins_noncloud[
                    i:
                ].sum()  # non-cloud pixels on the right of the thershold, included the boundary
            )

            # the optimal thershold is the one that can make the overlapping density as low as possible
            # i.e., the thershold is altered only when the error rate reduced 5% compared to the previous one
            if (
                num_errors_record - num_errors
            ) > threshold * num_errors_record:  # same as to (num_errors - num_errors_record)/num_errors_record < -threshold
                num_errors_record = num_errors.copy()
                thrd_record = thrd.copy()
        return over_rate, thrd_record
    return over_rate, None


def clear_probability(prob, clear):
    """Calculate the clear probability based on the given probability array and clear mask.

    Parameters:
    prob (numpy.ndarray): The probability array.
    clear (numpy.ndarray): The clear mask.

    Returns:
    float: The clear probability.

    """
    return np.percentile(prob[clear], C.HIGH_LEVEL)

# define functions inside
def shift_by_sensor(coords, height, view_zenith, view_azimuth, resolution):
    """
    Shifts the given coordinates based on the sensor parameters.

    Args:
        coords (numpy.ndarray): Array of coordinates to be shifted.
        height (float): Height of the cloud.
        view_zenith (1d array): Zenith angle of the sensor's view.
        view_azimuth (1d array): Azimuth angle of the sensor's view.
        resolution (float): Resolution of the sensor.

    Returns:
        numpy.ndarray: Array of shifted coordinates.
    """
    shift_dist = (
        height * np.tan(view_zenith) / resolution
    )  # in shifting pixels over the plate

    coords[:, 1] = coords[:, 1] + shift_dist * np.sin(view_azimuth)  # x-axis shift
    coords[:, 0] = coords[:, 0] - shift_dist * np.cos(view_azimuth) # y-axis shift
        
    # coords[:, 1] = coords[:, 1] + shift_dist * np.sin(view_azimuth)  # x-axis horizontal column
    # coords[:, 0] = coords[:, 0] - shift_dist * np.cos(view_azimuth)  # y_axis vertical  row
    # coords[:, 1] = coords[:, 1] + shift_dist * np.cos(view_azimuth)  # x-axis horizontal column
    # coords[:, 0] = coords[:, 0] + shift_dist * np.sin(view_azimuth)  # y_axis vertical  row
    
    # coords[:, 1] = coords[:, 1] + shift_dist * np.cos(
    #     np.pi / 2 - view_azimuth
    # )  # x-axis horizontal column
    # coords[:, 0] = coords[:, 0] + shift_dist * -np.sin(
    #     np.pi / 2 - view_azimuth
    # )  # y_axis vertical  row
    return coords


def shift_by_solar(coords, height, solar_elevation, solar_azimuth, resolution):
    """
    Shifts the given coordinates based on solar elevation and azimuth.

    Parameters:
    - coords (numpy.ndarray): Array of coordinates to be shifted.
    - height (float): Height of the cloud object.
    - solar_elevation (float): Solar elevation angle in radians.
    - solar_azimuth (float): Solar azimuth angle in radians.
    - resolution (float): Resolution of the image.

    Returns:
    - numpy.ndarray: Array of shifted coordinates.
    """
    shift_dist = (height / np.tan(solar_elevation)) / resolution  # in pixels
    coords[:, 1] = coords[:, 1] - shift_dist * np.sin(solar_azimuth)  # x-axis shift
    coords[:, 0] = coords[:, 0] + shift_dist * np.cos(solar_azimuth) # y-axis shift
    # coords[:, 1] = coords[:, 1] - shift_dist * np.cos(
    #     solar_azimuth
    # )  # x-axis horizontal column
    # coords[:, 0] = coords[:, 0] - shift_dist * np.sin(
    #     solar_azimuth
    # )  # y_axis vertical  row
    
    # coords[:, 1] = coords[:, 1] - shift_dist * np.cos(
    #     solar_azimuth - np.pi / 2
    # )  # x-axis horizontal column
    # coords[:, 0] = coords[:, 0] - shift_dist * np.sin(
    #     solar_azimuth - np.pi / 2
    # )  # y_axis vertical  row

    # if solar_azimuth < np.pi/2:
    #     coords[:, 1] = coords[:, 1] - shift_dist * np.sin(solar_azimuth)  # x-axis horizontal column
    #     coords[:, 0] = coords[:, 0] + shift_dist * np.cos(solar_azimuth)  # y_axis vertical  row
    # elif (solar_azimuth>=np.pi/2) and (solar_azimuth < np.pi):
    #     coords[:, 1] = coords[:, 1] - shift_dist * np.cos(solar_azimuth - np.pi/2) # x-axis
    #     coords[:, 0] = coords[:, 0] - shift_dist * np.sin(solar_azimuth - np.pi/2) # y_axis
    # elif (solar_azimuth>=np.pi) and (solar_azimuth < 3*np.pi):
    #     coords[:, 1] = coords[:, 1] + shift_dist * np.sin(solar_azimuth - np.pi) # x-axis
    #     coords[:, 0] = coords[:, 0] - shift_dist * np.cos(solar_azimuth - np.pi) # y_axis
    # else:
    #     coords[:, 1] = coords[:, 1] + shift_dist * np.cos(solar_azimuth - 3*np.pi/2) # x-axis
    #     coords[:, 0] = coords[:, 0] + shift_dist * np.sin(solar_azimuth - 3*np.pi/2) # y_axis
    
    # if solar_azimuth < np.pi:
    #     coords[:, 1] = coords[:, 1] - shift_dist * np.cos(
    #         solar_azimuth - np.pi / 2
    #     )  # x-axis horizontal column
    #     coords[:, 0] = coords[:, 0] - shift_dist * np.sin(
    #         solar_azimuth - np.pi / 2
    #     )  # y_axis vertical  row
    # else:
    #     coords[:, 1] = coords[:, 1] + shift_dist * np.cos(
    #         solar_azimuth - np.pi / 2
    #     )  # x-axis horizontal column
    #     coords[:, 0] = coords[:, 0] + shift_dist * np.sin(
    #         solar_azimuth - np.pi / 2
    #     )  # y_axis vertical  row
    return coords


def project_dem2plane4(ele, solar_elevation, solar_azimuth, resolution, mask_filled):
    """
    Projects a digital elevation model (DEM) to a plane based on solar elevation and azimuth.

    Args:
        ele (numpy.ndarray): The digital elevation model.
        solar_elevation (float): The solar elevation angle in degrees.
        solar_azimuth (float): The solar azimuth angle in degrees.
        resolution (float): The resolution of the DEM in meters.
        mask_filled (numpy.ndarray): A mask indicating filled pixels in the DEM.

    Returns:
        tuple: A tuple containing three arrays:
            - PLANE2IMAGE_ROW (numpy.ndarray): A matrix storing the mapping between the plane and the image (row indices).
            - PLANE2IMAGE_COL (numpy.ndarray): A matrix storing the mapping between the plane and the image (column indices).
            - PLANE_OFFSET (numpy.ndarray): The offset applied to the plane coordinates to make them positive.
    """

    ele = ele - np.percentile(
        ele[~mask_filled], 0.1
    )  # relative elevation 0.1 is to avoid the outlier
    ele[ele < 0] = 0 # minimum elevation is 0 after do the relative elevation
    # get the coordinates of all the dem pixels
    image_coords = np.argwhere(np.ones_like(ele, dtype=bool))
    # image_coords = np.indices(ele.shape).reshape(2, -1).T  # Faster than np.argwhere
    # projection along the solar direction
    plane_coords = shift_by_solar(
        image_coords.copy(), # do not vary the values
        ele[image_coords[:, 0], image_coords[:, 1]],
        np.deg2rad(solar_elevation).copy(), # convert to radiance
        np.deg2rad(solar_azimuth).copy(),  # convert to radiance
        resolution,
    )

    # create new array to preserve the plane_coords as positive
    PLANE_OFFSET = np.min(plane_coords, axis=0)
    plane_coords = plane_coords - PLANE_OFFSET  # make the plane_coords as positive
    PLANE_SHAPE = np.max(plane_coords, axis=0) + 1

    # create a matrix to store the mapping between the plane and the image
    PLANE2IMAGE_ROW = np.full(PLANE_SHAPE, -1, dtype=np.int32)
    PLANE2IMAGE_COL = np.full(PLANE_SHAPE, -1, dtype=np.int32)
    # convert to integer by round
    image_coords = np.round(image_coords).astype(np.int32)
    # append the mapping between the plane and the image
    PLANE2IMAGE_ROW[plane_coords[:, 0], plane_coords[:, 1]] = image_coords[:, 0]
    PLANE2IMAGE_COL[plane_coords[:, 0], plane_coords[:, 1]] = image_coords[:, 1]
    
    # since the round function may cause the same pixel to be mapped to different pixels, we need to fill the NaN values
    def fill_nan_nearest(plane_array, background=-1):
        """
        Fills NaN values in a 2D array using nearest valid neighbor values.

        Parameters:
        - plane_array: np.ndarray, 2D array with NaNs.

        Returns:
        - filled_array: np.ndarray, same shape as input but with NaNs filled.
        """
        nan_mask = plane_array == background
        
        # NaNs are filled
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, down, left, right
            shifted_array = np.roll(plane_array, shift=(dx, dy), axis=(0, 1))
            plane_array[nan_mask] = shifted_array[nan_mask]
        plane_array[plane_array == background] = 0  # Fill remaining NaNs with 0
        return plane_array

    # Apply to both PLANE2IMAGE_ROW and PLANE2IMAGE_COL
    PLANE2IMAGE_ROW = fill_nan_nearest(PLANE2IMAGE_ROW)
    PLANE2IMAGE_COL = fill_nan_nearest(PLANE2IMAGE_COL)
        
    return PLANE2IMAGE_ROW, PLANE2IMAGE_COL, PLANE_OFFSET


def project_dem2plane(ele, sensor_zenith, sensor_azimuth, solar_elevation, solar_azimuth, resolution):
    """
    Projects a Digital Elevation Model (DEM) onto a plane based on solar elevation and azimuth angles.
    uint16 is used for storing the plane coordinates to avoid overflow.
    Args:
        ele (np.ndarray): 2D array representing the elevation values of the DEM.
        solar_elevation (float): Solar elevation angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        resolution (float): Spatial resolution of the DEM.
    Returns:
        tuple: A tuple containing:
            - PLANE2IMAGE_ROW (np.ndarray): 2D array mapping plane coordinates to image row indices.
            - PLANE2IMAGE_COL (np.ndarray): 2D array mapping plane coordinates to image column indices.
            - PLANE_OFFSET (np.ndarray): Offset applied to plane coordinates to ensure they are positive.
    """

    # get the coordinates of all the dem pixels
    image_coords = np.argwhere(np.ones_like(ele, dtype=bool))
    # image_coords = np.indices(ele.shape).reshape(2, -1).T  # Faster than np.argwhere
    # Separate coordinates for odd and even indices for both row and column, in order to reduce it happens that mutiple image_coords for the same plane_coords (projected)
    image_coords_odd = image_coords[(image_coords[:, 0] % 2 == 1) & (image_coords[:, 1] % 2 == 1)]  # Odd rows and odd columns
    image_coords_even = image_coords[(image_coords[:, 0] % 2 == 0) & (image_coords[:, 1] % 2 == 0)]  # Even rows and even columns
    
    # adjust the shift by sensor
    
    shifts = []
    n_total = image_coords_odd.shape[0]
    for i in range(0, n_total, 1000000): # batch_size = 1000000
        batch = image_coords_odd[i:min(i + 1000000, n_total)]
        shifted = shift_by_sensor(batch.copy(),
                                  ele[batch[:, 0], batch[:, 1]],
                                  np.deg2rad(sensor_zenith[batch[:, 0], batch[:, 1]]), 
                                  np.deg2rad(sensor_azimuth[batch[:, 0], batch[:, 1]]), 
                                  resolution)
        shifts.append(shifted)
    image_coords_odd_shift = np.vstack(shifts)
    del shifts, batch

    # image_coords_odd_shift = shift_by_sensor(
    #     image_coords_odd.copy(),
    #     ele[image_coords_odd[:, 0], image_coords_odd[:, 1]],
    #     sensor_zenith[image_coords_odd[:, 0], image_coords_odd[:, 1]],
    #     sensor_azimuth[image_coords_odd[:, 0], image_coords_odd[:, 1]],
    #     resolution,
    # )
    
    # Projection along the solar direction for the first set of image coordinates
    plane_coords_odd = shift_by_solar(
        image_coords_odd_shift, # do not vary the values
        ele[image_coords_odd[:, 0], image_coords_odd[:, 1]],
        np.deg2rad(solar_elevation).copy(),  # Convert to radiance
        np.deg2rad(solar_azimuth).copy(),  # Convert to radiance
        resolution,
    )
    
    # # adjust the shift by sensor
    # image_coords_even_shift = shift_by_sensor(
    #     image_coords_even.copy(),
    #     ele[image_coords_even[:, 0], image_coords_even[:, 1]],
    #     sensor_zenith[image_coords_even[:, 0], image_coords_even[:, 1]],
    #     sensor_azimuth[image_coords_even[:, 0], image_coords_even[:, 1]],
    #     resolution,
    # )
    
    # adjust the shift by sensor
    shifts = []
    n_total = image_coords_even.shape[0]
    for i in range(0, n_total, 1000000): # batch_size = 1000000
        batch = image_coords_even[i:min(i + 1000000, n_total)]
        shifted = shift_by_sensor(batch.copy(), 
                                  ele[batch[:, 0], batch[:, 1]], 
                                  np.deg2rad(sensor_zenith[batch[:, 0], batch[:, 1]]), 
                                  np.deg2rad(sensor_azimuth[batch[:, 0], batch[:, 1]]), 
                                  resolution)
        shifts.append(shifted)
    image_coords_even_shift = np.vstack(shifts)
    del shifts, batch
    
    
    # Projection along the solar direction for the second set of image coordinates
    plane_coords_even = shift_by_solar(
        image_coords_even_shift, # do not vary the values
        ele[image_coords_even[:, 0], image_coords_even[:, 1]],
        np.deg2rad(solar_elevation).copy(),  # Convert to radiance
        np.deg2rad(solar_azimuth).copy(),  # Convert to radiance
        resolution,
    )
    
    # create new array to preserve the plane_coords as positive
    # Calculate PLANE_OFFSET before stacking
    # Get the minimum values for odd and even plane coordinates
    PLANE_OFFSET = np.minimum(np.min(plane_coords_odd, axis=0), np.min(plane_coords_even, axis=0))
    # Calculate PLANE_SHAPE using the maximum values across odd and even sets
    plane_coords_odd = plane_coords_odd - PLANE_OFFSET  # make the plane_coords as positive
    plane_coords_even = plane_coords_even - PLANE_OFFSET  # make the plane_coords as positive
    plane_coords_odd = plane_coords_odd.round().astype(np.uint16)
    plane_coords_even = plane_coords_even.round().astype(np.uint16)
    # Find the global maximum to define the plane shape
    PLANE_SHAPE = np.maximum(np.max(plane_coords_odd, axis=0), np.max(plane_coords_even, axis=0)) + 1

    # create a matrix to store the mapping between the plane and the image
    PLANE2IMAGE_ROW_ODD = np.full(PLANE_SHAPE, 0, dtype=np.uint16) 
    PLANE2IMAGE_ROW_EVEN = np.full(PLANE_SHAPE, 0, dtype=np.uint16)
    PLANE2IMAGE_COL_ODD = np.full(PLANE_SHAPE, 0, dtype=np.uint16)
    PLANE2IMAGE_COL_EVEN = np.full(PLANE_SHAPE, 0, dtype=np.uint16)
    # append the mapping between the plane and the image, first for the odd pixels
    PLANE2IMAGE_ROW_ODD[plane_coords_odd[:, 0], plane_coords_odd[:, 1]] = image_coords_odd[:, 0] # odd 
    PLANE2IMAGE_COL_ODD[plane_coords_odd[:, 0], plane_coords_odd[:, 1]] = image_coords_odd[:, 1] # odd 
    PLANE2IMAGE_ROW_EVEN[plane_coords_even[:, 0], plane_coords_even[:, 1]] = image_coords_even[:, 0] # even 
    PLANE2IMAGE_COL_EVEN[plane_coords_even[:, 0], plane_coords_even[:, 1]] = image_coords_even[:, 1] # even 
    return PLANE2IMAGE_ROW_ODD, PLANE2IMAGE_COL_ODD, PLANE2IMAGE_ROW_EVEN, PLANE2IMAGE_COL_EVEN, PLANE_OFFSET

def project_dem2plane2(ele, solar_elevation, solar_azimuth, resolution):
    """ Backup version of simple project_dem2plane function
    Projects a digital elevation model (DEM) to a plane based on solar elevation and azimuth.

    Args:
        ele (numpy.ndarray): The digital elevation model.
        solar_elevation (float): The solar elevation angle in degrees.
        solar_azimuth (float): The solar azimuth angle in degrees.
        resolution (float): The resolution of the DEM in meters.

    Returns:
        tuple: A tuple containing three arrays:
            - PLANE2IMAGE_ROW (numpy.ndarray): A matrix storing the mapping between the plane and the image (row indices).
            - PLANE2IMAGE_COL (numpy.ndarray): A matrix storing the mapping between the plane and the image (column indices).
            - PLANE_OFFSET (numpy.ndarray): The offset applied to the plane coordinates to make them positive.
    """

    # get the coordinates of all the dem pixels
    # image_coords = np.argwhere(np.ones_like(ele, dtype=bool))
    image_coords = np.indices(ele.shape).reshape(2, -1).T  # Faster than np.argwhere
    # projection along the solar direction
    plane_coords = shift_by_solar(
        image_coords.copy(), # do not vary the values
        ele[image_coords[:, 0], image_coords[:, 1]],
        np.deg2rad(solar_elevation).copy(), # convert to radiance
        np.deg2rad(solar_azimuth).copy(),  # convert to radiance
        resolution,
    )

    # create new array to preserve the plane_coords as positive
    PLANE_OFFSET = np.min(plane_coords, axis=0)
    plane_coords = plane_coords - PLANE_OFFSET  # make the plane_coords as positive
    PLANE_SHAPE = np.max(plane_coords, axis=0) + 1

    # create a matrix to store the mapping between the plane and the image
    PLANE2IMAGE_ROW = np.full(PLANE_SHAPE, 0, dtype=np.int32)
    PLANE2IMAGE_COL = np.full(PLANE_SHAPE, 0, dtype=np.int32)
    # convert to integer by round
    image_coords = np.round(image_coords).astype(np.int32)
    # append the mapping between the plane and the image
    PLANE2IMAGE_ROW[plane_coords[:, 0], plane_coords[:, 1]] = image_coords[:, 0]
    PLANE2IMAGE_COL[plane_coords[:, 0], plane_coords[:, 1]] = image_coords[:, 1]

    return PLANE2IMAGE_ROW, PLANE2IMAGE_COL, PLANE_OFFSET

def project_dem2plane3(ele, solar_elevation, solar_azimuth, resolution):
    # Back up for using combined code, with "code1*10000 + code2"
    """
    Projects a Digital Elevation Model (DEM) onto a plane based on solar elevation and azimuth angles.
    Args:
        ele (np.ndarray): 2D array representing the elevation values of the DEM.
        solar_elevation (float): Solar elevation angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        resolution (float): Spatial resolution of the DEM.
    Returns:
        tuple: A tuple containing:
            - PLANE2IMAGE_ROW (np.ndarray): 2D array mapping plane coordinates to image row indices.
            - PLANE2IMAGE_COL (np.ndarray): 2D array mapping plane coordinates to image column indices.
            - PLANE_OFFSET (np.ndarray): Offset applied to plane coordinates to ensure they are positive.
    """

    # get the coordinates of all the dem pixels
    # image_coords = np.argwhere(np.ones_like(ele, dtype=bool))
    image_coords = np.indices(ele.shape).reshape(2, -1).T  # Faster than np.argwhere
    # Separate coordinates for odd and even indices for both row and column, in order to reduce it happens that mutiple image_coords for the same plane_coords (projected)
    image_coords_odd = image_coords[(image_coords[:, 0] % 2 == 1) & (image_coords[:, 1] % 2 == 1)]  # Odd rows and odd columns
    image_coords_even = image_coords[(image_coords[:, 0] % 2 == 0) & (image_coords[:, 1] % 2 == 0)]  # Even rows and even columns
    
    # Projection along the solar direction for the first set of image coordinates
    plane_coords_odd = shift_by_solar(
        image_coords_odd.copy(), # do not vary the values
        ele[image_coords_odd[:, 0], image_coords_odd[:, 1]],
        np.deg2rad(solar_elevation).copy(),  # Convert to radiance
        np.deg2rad(solar_azimuth).copy(),  # Convert to radiance
        resolution,
    ).round().astype(np.int32)
    # Projection along the solar direction for the second set of image coordinates
    plane_coords_even = shift_by_solar(
        image_coords_even.copy(), # do not vary the values
        ele[image_coords_even[:, 0], image_coords_even[:, 1]],
        np.deg2rad(solar_elevation).copy(),  # Convert to radiance
        np.deg2rad(solar_azimuth).copy(),  # Convert to radiance
        resolution,
    ).round().astype(np.int32)
    # convert to integer by round
    # image_coords_odd = np.round(image_coords_odd).astype(np.int32) # max is 2,147,4 83,647
    # image_coords_even = np.round(image_coords_even).astype(np.int32) # max is 2,147,4 83,647
    # create new array to preserve the plane_coords as positive
    # Calculate PLANE_OFFSET before stacking
    # Get the minimum values for odd and even plane coordinates
    PLANE_OFFSET = np.minimum(np.min(plane_coords_odd, axis=0), np.min(plane_coords_even, axis=0))
    # Calculate PLANE_SHAPE using the maximum values across odd and even sets
    plane_coords_odd = plane_coords_odd - PLANE_OFFSET  # make the plane_coords as positive
    plane_coords_even = plane_coords_even - PLANE_OFFSET  # make the plane_coords as positive
    # Find the global maximum to define the plane shape
    PLANE_SHAPE = np.maximum(np.max(plane_coords_odd, axis=0), np.max(plane_coords_even, axis=0)) + 1

    # create a matrix to store the mapping between the plane and the image
    PLANE2IMAGE_ROW = np.full(PLANE_SHAPE, 0, dtype=np.int32)
    PLANE2IMAGE_COL = np.full(PLANE_SHAPE, 0, dtype=np.int32)
    # append the mapping between the plane and the image, first for the odd pixels
    PLANE2IMAGE_ROW[plane_coords_odd[:, 0], plane_coords_odd[:, 1]] = image_coords_odd[:, 0]
    PLANE2IMAGE_COL[plane_coords_odd[:, 0], plane_coords_odd[:, 1]] = image_coords_odd[:, 1]
    # encode the even row and column indices in the same matrix, 10000*loc_odd + loc_even, 10000 is enough for the image size of Landsat and Sentinel-2 (10/20m) (Sentinel2 image size is 10980*10980)
    PLANE2IMAGE_ROW[plane_coords_even[:, 0], plane_coords_even[:, 1]] = 10000*PLANE2IMAGE_ROW[plane_coords_even[:, 0], plane_coords_even[:, 1]] + image_coords_even[:, 0]
    PLANE2IMAGE_COL[plane_coords_even[:, 0], plane_coords_even[:, 1]] = 10000*PLANE2IMAGE_COL[plane_coords_even[:, 0], plane_coords_even[:, 1]] + image_coords_even[:, 1]
    return PLANE2IMAGE_ROW, PLANE2IMAGE_COL, PLANE_OFFSET

def segment_cloud_objects(cloud, min_area=3, buffer2connect=0, exclude=None, exclude_method = 'any'):
    """
    Segment cloud objects in the given cloud image.

    Parameters:
    - cloud: numpy.ndarray
        The cloud image to segment.
    - min_area: int, optional
        The minimum area (in pixels) for a cloud object to be considered.
    - exclude: numpy.ndarray
        The exclude base layer 
    - exclude_method: str, optional
        The method to exclude the cloud objects. Defaults to 'any'. or 'all'
        if any cloud pixels overlap with this exclude layer, the cloud will be excluded from the cloud_regions.
        if all cloud pixels overlap with this exclude layer, the cloud will be excluded from the cloud_regions.

    Returns:
    - cloud_objects: numpy.ndarray
        The labeled cloud objects.
    - cloud_regions: list of skimage.measure._regionprops.RegionProperties
        The region properties of the segmented cloud objects.
    """
    if (
        buffer2connect > 0
    ):  # connect neighbor clouds to match cloud shadows at same time
        cloud_dilated = utils.dilate(cloud, radius=buffer2connect)
        cloud_objects = label(
            cloud_dilated, background=0, return_num=False, connectivity=None
        )
        cloud_objects[~cloud] = 0
    else:
        cloud_objects = label(cloud, background=0, return_num=False, connectivity=None)

    cloud_regions = regionprops(cloud_objects)
    if min_area > 0:
        cloud_regions = [
            icloud for icloud in cloud_regions if icloud.area >= min_area
        ]  # filter out the very small clouds

    if exclude is not None:
        if exclude_method == 'any':
            cloud_regions = [
                icloud
                for icloud in cloud_regions
                if not any(exclude[icloud.coords[:, 0], icloud.coords[:, 1]])
            ]
        elif exclude_method == 'all':
            cloud_regions = [
                icloud
                for icloud in cloud_regions
                if not all(exclude[icloud.coords[:, 0], icloud.coords[:, 1]])
            ]
        # revise cloud_objects as well
        cloud_objects = np.zeros_like(cloud_objects) # initialize the cloud_objects as 0
        for icloud in cloud_regions:
            cloud_objects[icloud.coords[:, 0], icloud.coords[:, 1]] = icloud.label

    return cloud_objects, cloud_regions



def find_neighbor_cloud_base_height(icloud, num_close_clouds, cloud, 
                            record_cloud_base_heights, record_cloud_centroids,
                            cloud_height_min, cloud_height_max):
    """
    Determines the base height of a cloud based on nearby cloud objects.
    
    Parameters:
        icloud (int): Current cloud index.
        num_close_clouds (int): Minimum number of nearby cloud objects required.
        cloud (object): Cloud object with a centroid attribute.
        record_cloud_base_heights (numpy array): Array of recorded cloud base heights.
        record_cloud_centroids (numpy array): Array of recorded cloud centroids.
        cloud_height_min (float): Minimum allowable cloud height.
        cloud_height_max (float): Maximum allowable cloud height.
    
    Returns:
        float: Estimated cloud base height or 0.0 if conditions are not met.
    """
    if icloud >= num_close_clouds:
        # Find the closest cloud objects based on centroid distances
        distances = np.sum(np.abs(record_cloud_centroids[:icloud] - cloud.centroid), axis=1)
        sorted_indices = np.argsort(distances)
        close_cloud_heights = record_cloud_base_heights[sorted_indices]
        
        # Remove zero heights
        close_cloud_heights = close_cloud_heights[close_cloud_heights != 0]
        
        if len(close_cloud_heights) >= num_close_clouds:
            # Select the first num_close_clouds closest cloud heights
            close_cloud_heights = close_cloud_heights[:num_close_clouds]
            
            # Check the standard deviation of the selected heights
            if np.std(close_cloud_heights) >= 1000: # the height difference is too large see MFmask paper
                return 0.0  # Heights are too different
            else:
                record_close_cloud_base_height = np.percentile(close_cloud_heights, C.HIGH_LEVEL)
                
                # Validate height range
                if cloud_height_min <= record_close_cloud_base_height <= cloud_height_max:
                    return record_close_cloud_base_height
                else:
                    return 0.0
    
    return 0.0


def refine_cloud_thermal_properties(cloud_temp, cloud_radius, num_edge_pixels, 
    surface_temp_low, surface_temp_high, 
    cloud_height_min, cloud_height_max, 
    rate_dlapse, rate_dlapse_reduced
    ):
    """
    Refine the thermal properties of a cloud object based on its temperature, size, and height.
    Parameters:
    cloud_temp (ndarray): Array of cloud temperatures.
    cloud_radius (float): Radius of the cloud in pixels.
    num_edge_pixels (int): Number of edge pixels to consider for temperature adjustment.
    surface_temp_low (float): Lower bound of surface temperature.
    surface_temp_high (float): Upper bound of surface temperature.
    cloud_height_min (float): Minimum cloud height.
    cloud_height_max (float): Maximum cloud height.
    rate_dlapse (float): Temperature lapse rate for normal conditions.
    rate_dlapse_reduced (float): Reduced temperature lapse rate for colder clouds.
    Returns:
    tuple: A tuple containing:
        - cloud_temp (ndarray): Adjusted cloud temperature array.
        - cloud_temp_base (float): Base temperature of the cloud.
        - updated_cloud_height_min (float): Updated minimum cloud height.
        - updated_cloud_height_max (float): Updated maximum cloud height.
    """

    # Determine base temperature based on cloud size
    if cloud_radius > num_edge_pixels:  # Apply only to large clouds
        cloud_temp_base = np.percentile(
            cloud_temp, 
            100 * np.square(cloud_radius - num_edge_pixels) / np.square(cloud_radius)
        )
        # Adjust edge pixels to have the same value as cloud_temp_base
        cloud_temp[cloud_temp > cloud_temp_base] = cloud_temp_base
    else:
        cloud_temp_base = np.min(cloud_temp)

    # Adjust cloud height range for colder clouds
    # for some cloud deteciton models, like UNet and LightGBM, there are no clear sky land pixels to estimate the surface temperture condition, we will not narrow the range of heights
    if (surface_temp_low is not None) and (cloud_temp_base <= surface_temp_low):  
        updated_cloud_height_min = np.maximum(
            cloud_height_min, 
            (surface_temp_low - 4 - cloud_temp_base) / rate_dlapse
        )
        updated_cloud_height_max = np.minimum(
            cloud_height_max, 
            (surface_temp_high + 4 - cloud_temp_base) / rate_dlapse_reduced
        ) #  0.001
    else:
        updated_cloud_height_min = cloud_height_min
        updated_cloud_height_max = cloud_height_max

    return cloud_temp, cloud_temp_base, updated_cloud_height_min, updated_cloud_height_max

def match_cloud2shadow(
    cloud_regions,
    cloud_objects,
    pshadow,
    water,
    mask_filled,
    view_zenith,  # in degree
    view_azimuth,  # in degree
    solar_elevation,  # in degree
    solar_azimuth,  # in degree
    resolution,
    similarity=0.1,
    similarity_tolerance= 0.95,  # the tolerance of the similarity, in order to speed up the process
    penalty_weight = 0.9,  # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
    sampling_cloud=80000, # number of sampling pixels to find the shadow, in order to speed up the process. the value 0 means to use all the pixels
    thermal=None,
    surface_temp_low=None,
    surface_temp_high=None,
    ele=None,
    ele_base = 0,
    PLANE2IMAGE_ROW_ODD=None,
    PLANE2IMAGE_COL_ODD=None,
    PLANE2IMAGE_ROW_EVEN=None,
    PLANE2IMAGE_COL_EVEN=None,
    PLANE_OFFSET=None,
    apcloud=False,
):
    """
    Matches the cloud mask with the shadow mask to identify cloud shadows.
    
    Majoe modifications made, compared to Fmask 4.6:
    - All the similarity thresholds are 0.3, after testing 
    - Reduced sampling cloud to 60000 from 1000000, to speed up the process, but give higher weights to the pixels at the cloud boundary based on the pixel distance to the centroid
    - Adjusted the penalty weight to 0.1, to reduce the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds
    - In this case, shadow masking over water was taken back.
    
    Args:
        cloud_objects (ndarray): Binary cloud mask.
        pshadow (ndarray): shadow mask with weight.
        water (ndarray): Water mask, which is used to punish the pixels projected over water because water is easily identified as potential shadow area.
        mask_filled (ndarray): Filled mask.
        view_zenith (ndarray): View zenith angles in degrees.
        view_azimuth (ndarray): View azimuth angles in degrees.
        solar_elevation (float): Solar elevation angle in degrees.
        solar_azimuth (float): Solar azimuth angle in degrees.
        resolution (float): Resolution of the image.
        thermal (ndarray, optional): Thermal band data. Defaults to None.
        surface_temp_low (float, optional): Lower threshold for surface temperature. Defaults to None.
        surface_temp_high (float, optional): Upper threshold for surface temperature. Defaults to None.
        ele (ndarray, optional): Digital Elevation Model (DEM) data. Defaults to None.
        apcloud (ndarray, optional): Approve clouds identified by the cloud mask. Defaults to None.

    Returns:
        ndarray: Binary mask indicating the matched shadow areas.
    """
    # make sure the random sampling is the same fore reproducibility
    np.random.seed(C.RANDOM_SEED)
    pcloud = cloud_objects > 0
    # pshadow[pcloud] = 0  # exclude cloud from the potential shadow; # we like potential shadow, not cloud,
    # pcloud[pshadow] = 0 # when counnting the similarity, we need to exclude the shadow pixels from the cloud mask, since we may punish the pixels over clouds
    
    # check the thermal band is included
    thermal_included = thermal is not None
    # thermal_included = False # turn it off, in order to test the cloud shadow matching without thermal band. This is suitable for Sentinel-2.
    dem_proj  = ele is not None
    neighbor_height = True  # control the height of the cloud object by the neighbor cloud object

    # Note: the solar angle in radian, that we just use the scene-center angle, because at each pixel, the solar angle is not varied a lot
    # and, it will be time-comsuming if we read the solar angle for each pixel
    solar_elevation = np.deg2rad(solar_elevation)  # convert to radian
    solar_azimuth = np.deg2rad(solar_azimuth)  # convert to radian

    # max similarity between the cloud object and the cloud shadow object
    similarity_min = similarity  # thershold of approving shadows
    similarity_max = 0.95
    # similarity_buffer = similarity_tolerance #  0.9 #  0.95
    num_close_clouds = 14 # see MFask, Qiu et al., 2017, RSE 

    # number of inward pixes (90m) for cloud base temperature
    num_edge_pixels = int(90 / resolution)  # in pixels
    rate_elapse = 0.0065  # enviromental lapse rate 6.5 degrees/km in degrees/meter
    rate_dlapse = 0.0098  # dry adiabatic lapse rate 9.8 degrees/km   in degrees/meter
    rate_dlapse_reduced = 0.001  # a reduced wet adiabatic lapse rate of − 1 K km− 1

    # image size
    image_height, image_width = pshadow.shape
    shadow_mask_matched = np.zeros(pshadow.shape, dtype="bool")
    if apcloud: # activate this layer to store the approved cloud pixels only when the apcloud is true
        cloud_mask_matched = np.zeros(pshadow.shape, dtype="bool")
    else:
        cloud_mask_matched = None # just in case the cloud_mask_matched is not activated

    # height interval for finding the cloud shadow and move 2 pixel at a time
    cloud_height_interval = 2 * resolution * np.tan(solar_elevation) # in meters

    # search cloud shadow, starting from the cloud object closer to the center of the image
    # sort the cloud objects by the distance to the center of the image
    # in the center of the image, the cloud shadow is more likely to be found due to enough pixels to be matched
    centeroid_filled = np.array(regionprops((~mask_filled).astype(int))[0].centroid)
    cloud_regions = sorted(
        cloud_regions,
        key=lambda x: np.sum(np.square(centeroid_filled - x.centroid)),
    )

    # DEM projection
    if dem_proj:
        plane_shape = PLANE2IMAGE_ROW_ODD.shape # the shape of the plane projection, which is same as others
        cloud_surface_ele = np.percentile(ele[~mask_filled], 0.1) # 0.1 is to avoid the outlier
        
    if neighbor_height:
        # create a matrix to store the distance between the cloud objects
        record_cloud_centroids = np.array([cloud.centroid for cloud in cloud_regions])
        # cloud height
        record_cloud_base_heights = np.zeros(len(cloud_regions), dtype=np.float32)

    # record the cloud base height for the first cloud object
    # intergrate water into the filled layer, which will be punished if the shadow is projected over water
    mask_filled[water] = 1 # will be punished if the shadow is projected over water
    mask_filled[pcloud] = 1 # include the cloud pixels into the filled mask for sure, which will be punished by the shadow matching if overlapped with the other cloud objects
    mask_filled[pshadow] = 0 # exclude the shadow pixels from the filled mask for sure, which will not be punished by the shadow matching
    # pcloud[mask_filled] = 0  # exclude the filled pixels from the cloud mask for sure

    # iterate the cloud objects by enumate loop
    for icloud, cloud in enumerate(cloud_regions):
        # print('Cloud: ', icloud)
        # down-sampling the big cloud object
        if (sampling_cloud > 0) and (cloud.area > sampling_cloud):
            # total random sampling pixels
            csampling = np.random.choice(cloud.coords.shape[0], sampling_cloud, replace=False)
        else:
            csampling = np.arange(cloud.coords.shape[0])

        # copy the cloud coords since these information will be varied during the progress of matching cloud shadow
        cloud_coords = cloud.coords.copy()
        
        # in meters. 200m to 12km for cloud height usually
        cloud_height_min, cloud_height_max = 200.00 + ele_base, 12000.00 + ele_base # over the base elevation in the imagery
 
        # Calculate the cloud radius (assuming a circular cloud)
        cloud_radius = np.sqrt(cloud.area / (2 * np.pi))  # in pixels

        # narrow the cloud height range according to the thermal band if it is available
        if thermal_included:
            # Obtain thermal values for the cloud object
            cloud_temp = thermal[cloud_coords[:, 0], cloud_coords[:, 1]]

            cloud_temp, cloud_temp_base, cloud_height_min, cloud_height_max = refine_cloud_thermal_properties(
                cloud_temp, cloud_radius, num_edge_pixels, 
                surface_temp_low, surface_temp_high, 
                cloud_height_min, cloud_height_max, 
                rate_dlapse, rate_dlapse_reduced)
        
        # find the cloud height by the neighbor cloud object
        if neighbor_height and (icloud >= num_close_clouds):
            # will reset back to zero if the estimated cloud height is not valid within the range of cloud_height_min and cloud_height_max
            record_close_cloud_base_height = find_neighbor_cloud_base_height(icloud, num_close_clouds, cloud, 
                                record_cloud_base_heights, record_cloud_centroids,
                                cloud_height_min, cloud_height_max)
        else:
            record_close_cloud_base_height = 0.0  # zeros value will not start to find the cloud height nearby

        if dem_proj:
            # get the surface elevation underneath the cloud object
            # cloud_surface_ele = np.percentile(
            #    ele[cloud_coords[:, 0], cloud_coords[:, 1]], C.HIGH_LEVEL
            #)
            # Compared to Fmask 4.6, we use the mininum elevation of the cloud object to estimate the cloud surface elevation, since we have less commisison from snow/ice right now
            # cloud_surface_ele_max = np.percentile(
            #    ele[cloud_coords[:, 0], cloud_coords[:, 1]],  99.9 # 99.9 to avoid the DEM outliers
            # )
            # no need to use the percentile, since we have already used the 0.1 percentile to avoid the outliers
            cloud_surface_ele_max = np.max(ele[cloud_coords[:, 0], cloud_coords[:, 1]])
            
            # cloud must be higher than the surface elevation connected, for example, cloud around high mountains
            # convert cloud_surface_ele_max to float
            cloud_height_min = max(float(cloud_surface_ele_max), cloud_height_min)

        # recording variable for similarity between the cloud object and the cloud shadow
        record_similiarity = 0.0
        record_cloud_base_height = 0.0
        record_num_matched = 0

        # view angles for cloud object
        cloud_view_zenith = np.deg2rad(
            view_zenith[cloud_coords[:, 0], cloud_coords[:, 1]]
        )  # in radian
        cloud_view_azimuth = np.deg2rad(
            view_azimuth[cloud_coords[:, 0], cloud_coords[:, 1]]
        )  # in radian

        # iterate the cloud height from cloud_height_min to cloud_height_max
        for cloud_base_height in np.arange(
            cloud_height_min, cloud_height_max + cloud_height_interval, cloud_height_interval # + cloud_height_interval to make sure the cloud_height_max is included
        ):
            # when thermal available, create 3D cloud object with the cloud height according to the thermal band
            if thermal_included:
                cloud_height = (
                    cloud_temp_base - cloud_temp[csampling]
                ) * rate_elapse + cloud_base_height
            else:
                cloud_height = cloud_base_height # no 3D cloud object

            # make it as new variable to keep the original cloud object
            coords = cloud_coords[csampling]
            # calculate the cloud's coords
            coords = shift_by_sensor(
                coords,
                cloud_height,
                cloud_view_zenith[csampling],
                cloud_view_azimuth[csampling],
                resolution,
            )

            # when dem is available, adjust the base height of the cloud object relative to the reference plane
            if dem_proj:
                coords = shift_by_solar(
                    coords,
                    cloud_height + cloud_surface_ele,
                    solar_elevation,
                    solar_azimuth,
                    resolution,
                )  # relative to the reference plane
                # convert the coordinates from the plane to the image based on the plane_coords
                # find the coords within the plane_coords
                coords = coords - PLANE_OFFSET  # make the plane_coords as positive
                # Filter out coordinates that are outside the plane boundaries
                coords = coords[(
                    (coords[:, 0] >= 0) & (coords[:, 0] < plane_shape[0]) &   # row within bounds
                    (coords[:, 1] >= 0) & (coords[:, 1] < plane_shape[1])     # column within bounds
                )]  # remove the pixels out of the image (reference plane)

                # Stack the odd and even coordinates vertically
                coords = np.concatenate((np.hstack((PLANE2IMAGE_ROW_ODD[coords[:, 0], coords[:, 1]].reshape(-1, 1), PLANE2IMAGE_COL_ODD[coords[:, 0], coords[:, 1]].reshape(-1, 1))),
                                         np.hstack((PLANE2IMAGE_ROW_EVEN[coords[:, 0], coords[:, 1]].reshape(-1, 1), PLANE2IMAGE_COL_EVEN[coords[:, 0], coords[:, 1]].reshape(-1, 1)))), axis=0)

                # exclude the pixels with row and col are zero
                coords = coords[~((coords[:, 0] == 0) & (coords[:, 1] == 0))]  # remove the pixels that have not been mapped to the image, and projected to sample pixels on the plane
            else:
                coords = shift_by_solar(
                    coords, cloud_height, solar_elevation, solar_azimuth, resolution
                )

            # the id list of the pixels out of the image
            list_coords_outside = (
                (coords[:, 0] < 0) # equivalent to zero because some of the coords are not recored in the mapping matrix
                | (coords[:, 0] >= image_height)
                | (coords[:, 1] < 0) # equivalent to zero because some of the coords are not recored in the mapping matrix
                | (coords[:, 1] >= image_width)
            )
            coords = coords[~list_coords_outside]  # remove the pixels out of the image

            num_out_image = np.count_nonzero(list_coords_outside)

            # count the number of false pixels
            # the pixels over the cloud or shadow layer (exclude original cloud)
            shadow_projected = (
                cloud_objects[coords[:, 0], coords[:, 1]] != cloud.label
            )  # that include other clouds and clear pixels
            num_match2shadow = np.sum(
                shadow_projected * pshadow[coords[:, 0], coords[:, 1]]
            )  # here * used to consider multiply potential shadow layers, i.e, unet shadow plus filled shadow
            #num_match2cloud = np.count_nonzero(
            #    shadow_projected & pcloud[coords[:, 0], coords[:, 1]]
            #)  # here cloud_mask_binary has been merged with potential cloud and potential shadow together prior to
            num_match2filled = np.count_nonzero(
                shadow_projected & mask_filled[coords[:, 0], coords[:, 1]]
            )  # here mask_filled has been merged with potential cloud and potential shadow together prior to

            # simplify the similarity calculation with one line
            # 0.5 is used to punlish the projected pixels over cloud or filled layer, where there is no way to determine the potential cloud shadow:
            # num_match2shadow + 0.5 * (num_match2cloud + num_match2filled) + num_out_image
            # total number that is the total pixel (exclude original cloud): np.count_nonzero(shadow_projected) + num_out_image
            # C.EPS is used to avoid the division by zero
            # + num_out_image
            # similarity_matched = (num_match2shadow + (1.0 - penalty_weight) * (num_match2cloud + num_match2filled + num_out_image))/(np.count_nonzero(shadow_projected) + num_out_image + C.EPS)
            similarity_matched = (num_match2shadow + (1.0 - penalty_weight) * (num_match2filled + num_out_image))/(np.count_nonzero(shadow_projected) + num_out_image + C.EPS)

            # if the estimated height is over the record_close_cloud_base_height, we punish the similarity, with the ratio of the neigbor cloud height to the current cloud height
            # if the record_close_cloud_base_height is not zero, it means we have found the cloud shadow, in the previous clouds
            # if (record_close_cloud_base_height > 0) and (cloud_base_height > record_close_cloud_base_height):
            #     similarity_matched = similarity_matched * (
            #         record_close_cloud_base_height/cloud_base_height
            #     )

            # if we have found the cloud shadow, and the neighbor cloud height is lower than the previous one, then stop the iteration
            if not (
                record_num_matched > 0
                and record_close_cloud_base_height > 0 # when we use the neighbor cloud height information
                and record_close_cloud_base_height <= cloud_base_height
            ):
                # update the similarity recorded in this iteration
                if (
                    similarity_matched >= record_similiarity * similarity_tolerance # tolerance for the similarity with 0.95
                    and cloud_base_height < cloud_height_max - cloud_height_interval # not reach the maximum height
                    and record_similiarity < similarity_max # not reach the maximum similarity
                ):
                    if similarity_matched > record_similiarity:
                        record_similiarity = similarity_matched
                        record_cloud_base_height = cloud_base_height
                    continue  # continue the iteration
                else:
                    if record_similiarity >= similarity_min: #show check out the matched similarity is larger than the thershold
                        # a shadow was found
                        record_num_matched = record_num_matched + 1

                        # allow to continue to reach the height of the neighbor cloud object if the similarity is higher than the recorded one
                        if cloud_base_height < record_close_cloud_base_height:
                            if (
                                similarity_matched >= record_similiarity
                                # or similarity_matched >= similarity_max # within the tolerance of the similarity
                            ):
                                record_similiarity = similarity_matched
                                record_cloud_base_height = cloud_base_height
                            continue  # continue the iteration
                        # stop the iteration
                    else:
                        record_similiarity = 0.0  # reset the similarity if the value of similarity is too small
                        continue  # continue the iteration

            if record_num_matched == 0:
                break  # stop the iteration if no cloud shadow is found

            # if the code reaches here, it means we have found the cloud shadow finally
            # when thermal available, create 3D cloud object with the cloud height according to the thermal band
            if thermal_included:
                cloud_height = (
                    cloud_temp_base - cloud_temp  # use all the pixels
                ) * rate_elapse + record_cloud_base_height
            else:
                cloud_height = record_cloud_base_height
            # print('cloud_height: ', cloud_height)
            # print('record_cloud_base_height: ', record_cloud_base_height)
            # print('record_similiarity: ', record_similiarity)
            # calculate the cloud's coords
            coords = (
                cloud.coords
            )  # at the last time, we use this variable, so it is ok when it is altered without copy()
            # approved clouds
            if apcloud:
                cloud_mask_matched[coords[:, 0], coords[:, 1]] = True

            coords = shift_by_sensor(
                coords,
                cloud_height,
                cloud_view_zenith,
                cloud_view_azimuth,
                resolution,
            )

            # when dem is available, adjust the base height of the cloud object relative to the reference plane
            if dem_proj:
                coords = shift_by_solar(
                    coords,
                    cloud_height + cloud_surface_ele,
                    solar_elevation,
                    solar_azimuth,
                    resolution,
                )  # relative to the reference plane
                # convert the coordinates from the plane to the image based on the plane_coords
                # find the coords within the plane_coords
                coords = coords - PLANE_OFFSET  # make the plane_coords as positive
                coords = coords[(
                    (coords[:, 0] >= 0) & (coords[:, 0] < plane_shape[0]) &   # row within bounds
                    (coords[:, 1] >= 0) & (coords[:, 1] < plane_shape[1])     # column within bounds
                )]  # remove the pixels out of the image (reference plane)
      
                # Decode the row and column indices for PLANE2IMAGE_ROW and PLANE2IMAGE_COL, and stack odd and even coordinates together
                # Fetch the values from the mapping arrays once
                # plane2image_row_vals = PLANE2IMAGE_ROW[coords[:, 0], coords[:, 1]]
                # plane2image_col_vals = PLANE2IMAGE_COL[coords[:, 0], coords[:, 1]]
                # # Compute the odd and even parts separately
                # row_odd, row_even = divmod(plane2image_row_vals, 10000)
                # col_odd, col_even = divmod(plane2image_col_vals, 10000)
                # Stack the results efficiently
                # row_odd, row_even = PLANE2IMAGE_ROW[coords[:, 0], coords[:, 1], :]
                # col_odd, col_even = PLANE2IMAGE_COL[coords[:, 0], coords[:, 1], :]
                # Stack the results efficiently
                # coords = np.column_stack((np.concatenate((row_odd, row_even)),
                #                        np.concatenate((col_odd, col_even))))
                
                # Decode the row and column indices for PLANE2IMAGE_ROW and PLANE2IMAGE_COL, and stack odd and even coordinates together
                # coords = np.vstack((np.column_stack((PLANE2IMAGE_ROW[coords[:, 0], coords[:, 1]] // 10000, PLANE2IMAGE_COL[coords[:, 0], coords[:, 1]] // 10000)) , 
                #                     np.column_stack((PLANE2IMAGE_ROW[coords[:, 0], coords[:, 1]] % 10000 , PLANE2IMAGE_COL[coords[:, 0], coords[:, 1]] % 10000))))
                
                # Stack the odd and even coordinates vertically
                coords = np.concatenate((np.hstack((PLANE2IMAGE_ROW_ODD[coords[:, 0], coords[:, 1]].reshape(-1, 1), PLANE2IMAGE_COL_ODD[coords[:, 0], coords[:, 1]].reshape(-1, 1))),
                                         np.hstack((PLANE2IMAGE_ROW_EVEN[coords[:, 0], coords[:, 1]].reshape(-1, 1), PLANE2IMAGE_COL_EVEN[coords[:, 0], coords[:, 1]].reshape(-1, 1)))), axis=0)
                # exclude the pixels with row and col are zero
                coords = coords[~((coords[:, 0] == 0) & (coords[:, 1] == 0))]  # remove the pixels that have not been mapped to the image or projected to sample pixels on the plane
            else:
                coords = shift_by_solar(
                    coords, cloud_height, solar_elevation, solar_azimuth, resolution
                )
            list_coords_outside = (
                (coords[:, 0] < 0) # equivalent to zero because some of the coords are not recored in the mapping matrix
                | (coords[:, 0] >= image_height)
                | (coords[:, 1] < 0) # equivalent to zero because some of the coords are not recored in the mapping matrix
                | (coords[:, 1] >= image_width)
            )
            coords = coords[
                ~list_coords_outside
            ]  # remove the pixels out of the image

            # recording
            shadow_mask_matched[coords[:, 0], coords[:, 1]] = True
            if (
                cloud_radius > num_edge_pixels
            ):  # not for small cloud object, its height will be used to assign the cloud height
                record_cloud_base_heights[icloud] = cloud_base_height
            # record_num_matched = record_num_matched + 1
            # stop the iteration
            break
    # exclude cloud pixels
    shadow_mask_matched[cloud_objects > 0] = False
    return shadow_mask_matched, cloud_mask_matched


class Physical:
    """Physical model for cloud detection"""

    image = None
    activated = None # indicate it was initiated or not, and then turn to False or True
    pcp = None
    lprob_var = None
    lprob_temp = None
    wprob_temp = None
    wprob_bright = None
    prob_cirrus = None
    water = None
    snow = None
    surface_temp_low = None  # only when the physical model was activated, we can get the surface temperature, that will be used to narrow the height of the cloud
    surface_temp_high = None
    threshold_shift = 0.0  # a global threshold shift for all the cloud probabilities, which is used to adjust the sensitivity of the cloud detection

    # options to control the cloud probabilities
    # and by default, all options are True
    options = [True, True, True]
    options_var = [True, False]
    options_temp = [True, False]
    options_cirrus = [True, False]

    # we do not use this option right now, since we do not want to change the physical rules
    # erosion of water mask, which is used to remove the small water pixels, such as narrow river and ponds. For those, we do not need to get based on the gswo dataset, just use the spectral test
    # after testing, we only used the water_erosion_radius
    swo_erosion_radius = 0 # minus "-" means to erosion unit is meters, and the positive value means the unit is pixels; which was used to test different units; 0 will not change the water mask
    # default value is 965 meters, which the 95th percentile of global river widths recorded in GRWL v1.1 dataset (Allen and Pavelsky, 2018, Science)
    water_erosion_radius = 0 # minus "-" means to erosion unit is meters, and the positive value means the unit is pixels; which was used to test different units (swo + spectral test); this only is used to calculate the cloud probability, not the final water mask.

    # the minimum number of clear pixels to start up the cloud probability model, as well as the minimum number for representing clear surface pixels, which is used to normalize the thermal band
    min_clear = 40000
    # minimum number of cloud pixels to start up the cloud shadow matching
    sampling_cloud = 80000
    # min similarity between the cloud object and the cloud shadow object
    similarity = 0.1
    # the tolerance of the similarity, in order to speed up the process
    similarity_tolerance = 0.95
    # the penalty for the cloud shadow matching to the regions that we do not understand the underlanding surface, like out-of-image and other identified clouds, as well as water pixels.
    # the water pixels are included here because they are easily identified as potential shadow area in land imagery
    penalty_weight = 0.9
    
    @property
    def abs_clear(self):
        """clear pixels

        Returns:
            2d array in bool: True for clear pixels, including land and water
        """
        return np.logical_and(
            self.image.obsmask, ~self.pcp
        )  # relying one the image object, we can get the obsmask

    @property
    def abs_clear_land(self):
        """
        Determine clear land pixels.
            This method identifies clear land pixels by checking for non-water and clear conditions.
            if the number of clear land pixels is less than the minimum required, then it falls back to using the original clear land pixels.
            Args:
                np.ndarray: A 2D boolean array where True indicates clear land pixels.
        """
        _abs_clear_land = np.logical_and(~self.water, self.abs_clear) # do not need use the eroded water
        if (
            np.count_nonzero(_abs_clear_land) < self.min_clear
        ):  # in case we do not have enought clear land pixels
            _abs_clear_land = self.abs_clear
        return _abs_clear_land
    

    @property
    def abs_clear_water(self):
        """clear water pixels

        Returns:
            2d array in bool: True for clear water pixels
        """
        return np.logical_and(self.water, self.abs_clear)

    @property
    def prob_variation(self):
        """
        Calculate the cloud probability variation for land and water

        Returns:
            float: The cloud probability variation.
            None: If the method is not activated.
        """
        if self.activated:
            # record the orginal options
            options_var_temp = self.options_var.copy()
            options_temp_temp = self.options_temp.copy()
            options_cirrus_temp = self.options_cirrus.copy()

            # generate the cloud probability for temperature
            self.options_var = [False]
            self.options_temp = [True]
            self.options_cirrus = [False]
            prob_temp, _, _ = self.select_cloud_probability(adjusted=False)

            # generate the cloud probability for spectral variation
            self.options_var = [True]
            self.options_temp = [True]
            self.options_cirrus = [False]
            prob_var, _, _ = self.select_cloud_probability(adjusted=False)
            prob_var = prob_var / (
                prob_temp + C.EPS
            )  # select_cloud_prob does not support the rule only var

            # restore the orginal options
            self.options_var = options_var_temp.copy()
            self.options_temp = options_temp_temp.copy()
            self.options_cirrus = options_cirrus_temp.copy()

            return prob_var
        return None

    @property
    def prob_temperature(self):
        """
        Calculate the cloud probability temperature for land and water

        Returns:
            float: The cloud probability temperature.
            None: If the method is not activated.
        """
        if self.activated:
            # record the orginal options
            options_var_temp = self.options_var.copy()
            options_temp_temp = self.options_temp.copy()
            options_cirrus_temp = self.options_cirrus.copy()

            # generate the cloud probability for temperature
            self.options_var = [False]
            self.options_temp = [True]
            self.options_cirrus = [False]
            prob_temp, _, _ = self.select_cloud_probability(adjusted=False)

            # restore the orginal options
            self.options_var = options_var_temp.copy()
            self.options_temp = options_temp_temp.copy()
            self.options_cirrus = options_cirrus_temp.copy()

            return prob_temp
        else:
            return None

    @property
    def prob_cloud(self):
        """Calculate the cloud probability.

        Returns:
            float: The cloud probability layer with the recorded options.
            None: If the method is not activated.
        """
        if self.activated:
            cloud_prob, _, _ = self.select_cloud_probability(
                seed=None,
                options_var=[self.options[0]],
                options_temp=[self.options[1]],
                options_cirrus=[self.options[2]],
                adjusted=True,
            )
            return cloud_prob
        else:
            return None

    @property
    def cloud(self):
        """Calculate the cloud mask based on the cloud probability.

        Returns:
            2d array in bool: The cloud mask.
        """
        if self.activated:
            return self.prob_cloud > self.threshold
        return None
    
    @property
    def cold_cloud(self):
        """
        Determines if a pixel is classified as a extremly cold cloud.

        Returns:
            bool: True if the pixel is classified as a cold cloud, False otherwise.
        """
        if (self.activated and self.image.data.exist("tirs1")):
            return self.image.data.get("tirs1") < (self.surface_temp_low - self.threshold_cold_cloud) # in degree
        return None

    def set_options(self, options_var=[True, False], options_temp=[True, False], options_cirrus=[True, False]):
        """set the options for cloud probabilities

        Args:
            options_var (list of bool): variation options
            options_temp (list of bool): temporal options
            options_cirrus (list of bool): cirrus options
        """
        self.options_var = copy.deepcopy(options_var)
        self.options_temp = copy.deepcopy(options_temp)
        self.options_cirrus = copy.deepcopy(options_cirrus)

    def init_constant_filter(self):
        """
        Initializes simple masks for water and snow and other base information for cloud shadow, which can be used for pure lightgbm and unet models.
        This method creates various probability layers based on the datacube of the image.
        The layers include activation status, cloud probability, land probability, water probability,
        snow probability, cirrus cloud probability, and surface temperature ranges.
        The `compute_cloud_probability_layers` function is used to generate these layers, with 
        `min_clear` set to infinity to ensure that the cloud probability layers are not activated.
        Attributes:
            activated (ndarray): Activation status of the probability layers.
            pcp (ndarray): Cloud probability layer.
            lprob_var (ndarray): Land probability layer (variable).
            lprob_temp (ndarray): Land probability layer (temporal).
            wprob_temp (ndarray): Water probability layer (temporal).
            wprob_bright (ndarray): Water probability layer (brightness).
            prob_cirrus (ndarray): Cirrus cloud probability layer.
            water (ndarray): Water probability layer.
            snow (ndarray): Snow probability layer.
            surface_temp_low (ndarray): Lower bound of surface temperature.
            surface_temp_high (ndarray): Upper bound of surface temperature.
        """
        
        # create the probability layers according to the datacube
        (
            self.activated,
            self.pcp,
            self.lprob_var,
            self.lprob_temp,
            self.wprob_temp,
            self.wprob_bright,
            self.prob_cirrus,
            self.water,
            self.snow,
            self.surface_temp_low,  # only when the physical model was activated, we can get the surface temperature, that will be used to narrow the height of the cloud
            self.surface_temp_high,
        ) = compute_cloud_probability_layers(self.image, min_clear=np.inf, swo_erosion_radius = self.swo_erosion_radius, water_erosion_radius=self.water_erosion_radius)
        # setup min_clear is a max value, making the cloud probability layers not to be activated, and then just do the simplest 

    def init_cloud_probability(self) -> None:
        """
        Generates the cloud probability layers based on the datacube.

        Returns:
            None
        """
        # create the probability layers according to the datacube
        (
            self.activated,
            self.pcp,
            self.lprob_var,
            self.lprob_temp,
            self.wprob_temp,
            self.wprob_bright,
            self.prob_cirrus,
            self.water,
            self.snow,
            self.surface_temp_low,  # only when the physical model was activated, we can get the surface temperature, that will be used to narrow the height of the cloud
            self.surface_temp_high,
        ) = compute_cloud_probability_layers(self.image, min_clear=self.min_clear, swo_erosion_radius = self.swo_erosion_radius, water_erosion_radius=self.water_erosion_radius)

    def select_cloud_probability(
        self,
        seed=None,
        label_cloud=1,
        label_noncloud=0,
        options_var=None,
        options_temp=None,
        options_cirrus=None,
        adjusted=True,
        show_figure=False,
    ):
        """Selects the cloud probability based on the given parlayer of seed with cloud and noncloud

        Args:
            seed (optional): The seed used for random number generation. Defaults to None.
            label_cloud (optional): The label for cloud pixels. Defaults to 1.
            label_noncloud (optional): The label for non-cloud pixels. Defaults to 0.
            options_var (optional): The boolean list of "yes" or "no" to use this. Defaults to None.
            options_temp (optional): The boolean list of "yes" or "no" to use this. Defaults to None.
            options_cirrus (optional): The boolean list of "yes" or "no" to use this. Defaults to None.
            adjusted (optional): The boolean to adjust the threshold. Defaults to True.

        Returns:
            mask_prob_final: The final mask probability.
            options: The selected options for variation, brightness, and cirrus.
            thrd_opt: The optimal threshold value.
        """
        # mask out the absolute clear pixels for land and water pixels
        mask_absclear_land = self.abs_clear_land
        mask_absclear_water = self.abs_clear_water

        # Test the optimal model by histogram overlapped
        # Convert as exporting cloud probability
        prob_record = None
        ol_record = 1.0
        options_record = None
        threshold_record = 0
        # at default, we use the current options
        if options_var is None:
            options_var = self.options_var
        if options_temp is None:
            options_temp = self.options_temp
        if options_cirrus is None:
            options_cirrus = self.options_cirrus

        # in case no cirrus prob. e.g., Landsat 4-7
        if self.woc == 0:
            options_cirrus = [False]

        bin_width = 0.025  # i.e., scale the cloud probability to 0-1 with 0.025
        for cir in options_cirrus:  # cirrus band has the highest priority
            for tmp in options_temp:  # at the second priority
                for var in options_var:  # variation and brightness prob.
                    # skip the option, when we do not use any components
                    if (not var) and (not tmp) and (not cir):
                        continue
                    # skip the option, only var is used
                    if (var) and (not tmp) and (not cir):
                        continue
                    # when the perfect model was created at the past round
                    if ol_record == 0:
                        continue
                    # check prob. over land that we have to use the previous prob. for land
                    mask_prob = combine_cloud_probability(
                        var,
                        tmp,
                        cir,
                        self.lprob_var,
                        self.lprob_temp,
                        self.prob_cirrus,
                        self.woc,
                        mask_absclear_land,
                        adjusted=adjusted,
                    )
                    # check prob. over water
                    if np.any(mask_absclear_water):
                        # in case when only temporial prob. is used (actually we do not have for water in Sentinel-2), we use the previous prob. for water
                        if (
                            (not var)
                            & (not cir)
                            & tmp
                            & (not isinstance(self.wprob_temp, np.ndarray))
                        ):
                            # pylint: disable=unsubscriptable-object
                            if prob_record is not None:
                                mask_prob = np.where(
                                    np.bitwise_and(self.water, prob_record > mask_prob),
                                    prob_record,
                                    mask_prob,
                                )
                        else:
                            mask_prob_water = combine_cloud_probability(
                                var,
                                tmp,
                                cir,
                                self.wprob_bright,
                                self.wprob_temp,
                                self.prob_cirrus,
                                self.woc,
                                mask_absclear_water,
                                adjusted=adjusted,
                            )
                            # only update the pixels where the prob. over water is higher than the prob. over land
                            # which is the same as the previous Fmask, where the logistical math 'or' is used.
                            # to mask the thin cloud over the water just.
                            mask_prob = np.where(
                                np.bitwise_and(self.water, mask_prob_water > mask_prob),
                                mask_prob_water,
                                mask_prob,
                            )
                    if seed is not None:
                        # convert the cloud probability to the seed groups with 1 array
                        seed_cloud_prob, seed_noncloud_prob, prob_range = (
                            convert2seedgroups(
                                mask_prob,
                                seed,
                                label_cloud,
                                label_noncloud,
                                bin_width=bin_width,
                                equal_num=False,
                            )
                        )
                        # get the overlap rate between cloud and non-cloud pixels
                        ol, opt_thrd = overlap_cloud_probability(
                            seed_cloud_prob,
                            seed_noncloud_prob,
                            prob_range=prob_range,
                            prob_bin=bin_width,
                        )
                        if C.MSG_FULL:
                            print(
                                f">>> cloud probability ({str(var)[0]}{str(tmp)[0]}{str(cir)[0]}) | overlap: {ol:.3f} | optimal threshold: {opt_thrd:.3f}"
                            )
                        # ol, thrd_opt = overlap_cloud_probability(mask_prob, mask_seed, label_cloud=1, label_noncloud=0, prob_range = prob_range, prob_bin=0.025)
                        if show_figure:
                            # make title with the options at the end (TTT)
                            utils.show_cloud_probability(
                                mask_prob,
                                self.image.filled,
                                title=f"Cloud probability ({str(var)[0]}{str(tmp)[0]}{str(cir)[0]})",
                            )
                            # show the histogram of the cloud probability
                            utils.show_cloud_probability_hist(
                                seed_cloud_prob,
                                seed_noncloud_prob,
                                prob_range,
                                prob_bin=bin_width,
                                title=f"Cloud probability ({str(var)[0]}{str(tmp)[0]}{str(cir)[0]})",
                            )
                            
                            # show the cloud mask
                            mask_cloud = np.zeros(self.image.shape, dtype="uint8")
                            mask_cloud[mask_prob < opt_thrd]= label_noncloud
                            mask_cloud[mask_prob >= opt_thrd] = label_cloud
                            mask_cloud[self.image.filled] = 2 # filled
                            utils.show_cloud_mask(
                                mask_cloud, ["noncloud", "cloud", "filled"], "Cloud mask"
                            )
                    else:
                        ol = 1.0 # 100% overlap between cloud and noncloud pixels
                        opt_thrd = self.threshold # default threshold
                    # update the optimal model if the overlap rate is decreased
                    if (ol_record == 1) or (
                        ((ol_record - ol) / (ol_record + C.EPS)) > self.overlap
                    ):  # 2 % decreased
                        ol_record = ol  # that cannot use .copy()
                        prob_record = mask_prob.copy()
                        options_record = [var, tmp, cir]
                        threshold_record = opt_thrd
                        

        if (
            seed is not None
        ):  # only for the seed pixels which are used to find the optimal threshold
            self.options = options_record  # update the options determined
            self.threshold = threshold_record  # update the threshold determined
            if C.MSG_FULL:
                print(
                    f">>> optimal cloud probability ({str(options_record[0])[0]}{str(options_record[1])[0]}{str(options_record[2])[0]}) | optimal threshold: {threshold_record:.3f}"
                )

        return prob_record, options_record, threshold_record

    def match_cloud2shadow(
        self,
        cloud_objects,
        cloud_regions,
        pshadow,
        water,
        thermal_adjust=True,
    ):
        """
        Match shadows by identified clouds.

        Args:
            cloud_objects (ndarray): Binary cloud object mask.
            cloud_regions (list): List of cloud regions.
            pshadow (ndarray, number): Potential shadow layer.
            water (ndarray): Water mask.

        Returns:
            tuple: A tuple containing the projected cloud shadows and the updated cloud layer.
        """
        # convert to relative elevation to the mininum elevation within the imagery
        dem_relative = self.image.data.get("dem")
        dem_base = np.percentile(dem_relative[self.image.obsmask], 0.1)  # relative elevation 0.1 is to avoid the outlier
        dem_relative = dem_relative - dem_base

        dem_relative[dem_relative < 0] = 0 # minimum elevation is 0 after do the relative elevation
        sensor_zenith = self.image.read_angle("SENSOR_ZENITH", unit="degree")
        sensor_azimuth = self.image.read_angle("SENSOR_AZIMUTH", unit="degree")
        # project the dem to the reference plane, at the first step
        plane2image_row_odd, plane2image_col_odd, plane2image_row_even, plane2image_col_even, plane_offset = project_dem2plane(
            dem_relative,
            sensor_zenith,
            sensor_azimuth,
            self.image.sun_elevation,
            self.image.sun_azimuth,
            self.image.resolution,
        )
        
        if C.MSG_FULL:
            print(">>> matching cloud shadows")
        
        # only when we have the data for thermal band, we can use thermal band to narrow the height of the cloud and estimate 3D cloud object
        if thermal_adjust:
            # get the thermal band data
            if self.image.data.exist("tirs1"):
                thermal = self.image.data.get("tirs1").copy()
            else:
                thermal = None
        else:
            thermal = None
        self.image.clean_data() # clean the data to save memory
        # call the function defined outside the class
        shadow_last, _ = match_cloud2shadow(
            cloud_regions,
            cloud_objects,
            pshadow,
            water,
            self.image.filled.copy(), # copy it, since we will modify it with adding water layer to punish the shadow over water
            sensor_zenith,
            sensor_azimuth,
            self.image.sun_elevation,
            self.image.sun_azimuth,
            self.image.resolution,
            similarity=self.similarity,
            similarity_tolerance=self.similarity_tolerance,
            penalty_weight=self.penalty_weight,
            sampling_cloud=self.sampling_cloud,
            thermal=thermal,
            surface_temp_low=self.surface_temp_low,
            surface_temp_high=self.surface_temp_high,
            ele=dem_relative,
            ele_base=dem_base,
            PLANE2IMAGE_ROW_ODD=plane2image_row_odd,
            PLANE2IMAGE_COL_ODD=plane2image_col_odd,
            PLANE2IMAGE_ROW_EVEN=plane2image_row_even,
            PLANE2IMAGE_COL_EVEN=plane2image_col_even,
            # PLANE2IMAGE_ROW=plane2image_row,
            # PLANE2IMAGE_COL=plane2image_col,
            PLANE_OFFSET=plane_offset,
            apcloud=False,
        )
        return shadow_last

    def __init__(self, predictors, woc, threshold, overlap=0.0) -> None:
        """Initialize the physical model"""
        # only when the physical model was activated, we can get the surface temperature, that will be used to narrow the height of the cloud
        self.image = None
        # the predictors for the physical model
        self.predictors = predictors
        # weight of cirrus probability
        self.woc = woc
        # threshold to separate cloud and non-cloud pixels
        self.threshold = threshold
        # the overlap density between cloud and non-cloud pixels to move further
        self.overlap = overlap  # 0% overlap increasing compared to the previous test to alter the physical models
        # extremely cold cloud
        self.threshold_cold_cloud = 35  # in degree
