"""
Author: Shi Qiu
Email: shi.qiu@uconn.edu
Date: 2024-05-24
Version: 1.0.0
License: MIT

Description:
This script defines the Satellite class for reading satellite data from Landsat and Sentinel-2 images.

Changelog:
- 1.0.0 (2024-05-24): Initial release.
"""

# Reference:
# see Landsat Collection 2 Data Dictionary
# https://www.usgs.gov/centers/eros/science/landsat-collection-2-data-dictionary (Last access on 7/1/2023)
# see Landsat band designations
# https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites (Last access on 7/1/2023)
# see conversion from DN to TOA
# https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product (Last access on 7/1/2023)
# see Radiometric Saturation Quality Assessment band
# https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands (Last access on 7/1/2023)

# pylint: disable=line-too-long
# pylint: disable=invalid-name
# pylint: too-many-branches
# pylint: too-many-statements
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from skimage.transform import resize
import pyproj
from bitlib import BitLayer
from utils import (
    hot,
    whiteness,
    ndvi,
    ndsi,
    ndbi,
    sfdi,
    cdi,
    variation,
    gen_dem,
    gen_gswo,
    normalize_datacube,
    fill_nan_wise
)
import constant as C


class Data:
    """
    A class representing satellite data.

    Attributes:
        data (ndarray): The satellite data.
        bands (list): The list of bands in the satellite data.

    Methods:
        get(band): Returns the data for the specified band.

    """

    def exist(self, band):
        """
        Check if a band exists in the satellite object.

        Parameters:
        - band (str): The name of the band to check.

        Returns:
        - bool: True if the band exists, False otherwise.
        """
        return band in self.bands

    def get(self, band):
        """
        Returns the data for the specified band.

        Args:
            band (str): The name of the band.

        Returns:
            ndarray: The data for the specified band.

        """
        if isinstance(band, list):
            return self.data[[self.bands.index(pred) for pred in band], :, :]
        else:
            if band in self.bands:
                return self.data[self.bands.index(band), :, :]
            else:
                return None
    
    def copyget(self, band):
        """
        Returns the data for the specified band by deep copy(), in array, xx.copy()

        Args:
            band (str): The name of the band.

        Returns:
            ndarray: The data for the specified band.

        """
        if isinstance(band, list):
            return self.data[[self.bands.index(pred) for pred in band], :, :].copy()
        else:
            if band in self.bands:
                return self.data[self.bands.index(band), :, :].copy()
            else:
                return None

    def __init__(self, data, bands):
        """
        Initializes a new instance of the Data class.

        Args:
            data (ndarray): The satellite data.
            bands (list): The list of bands in the satellite data.

        """
        self.data = data
        self.bands = bands


class Satellite:
    """Base class for satellite data object"""

    def __init__(self, folder):
        self.folder: str = folder
        self.name: str = Path(folder).stem
        self.spacecraft: str = None
        self.destination: str = self.folder
        self.metafile: str = None
        self.resolution: int = None
        self.bands: list = None  # band name
        self.data: Data = None
        self.saturation: BitLayer = None
        self.obsmask: np.ndarray = None
        self.profile = None
        self.profilefull = None
        self.lat_north: float = None
        self.lat_south: float = None
        self.lat_center: float = None
        self.lon_center: float = None
        self._filled_cache = None
        self._sensor_zenith = None # for Sentinel-2's cube angle 13 by 22 by 22
        self._sensor_azimuth = None # for Sentinel-2's cube angle 13 by 22 by 22

    # %% Properties
    @property
    def filled(self):
        """mask of filled pixel

        Returns:
            2d array (bool): mask of filled pixel
        """
        if self._filled_cache is None:
            # pylint: disable=invalid-unary-operand-type
            self._filled_cache = ~self.obsmask
        return self._filled_cache

    def clear_cache(self):
        """clear the caches of the object to release the memory"""
        self._filled_cache = None

    @property
    def obsnum(self):
        """number of observation

        Returns:
            int: number of observation
        """
        return np.count_nonzero(self.obsmask)

    @property
    def shape(self):
        """shape of the image

        Returns:
            tuple: shape of the image
        """
        return self.obsmask.shape

    def normalize(self, **kwargs):
        """
        Normalize the data in the datacube.

        Args:
            percentiles (list, optional): List of two percentiles used for normalization. Default is [1, 99].
            range (list, optional): List of two values specifying the desired range for normalization. Default is [-1, 1].

        Returns:
            None
        """
        percentiles = kwargs.get("percentiles", [1, 99])
        srange = kwargs.get("srange", [-1, 1])
        self.data.data = normalize_datacube(self.data.data, obsmask=self.obsmask, percentiles=percentiles, srange=srange)

    @property
    def tile(self):
        """tile of the image

        Returns:
            tuple: tile of the image
        """
        raise NotImplementedError("Property must be implemented in subclass")

    # %%  Methods implemented in the class
    def get_spacecraft_bands(self, spacecraft):
        """Get default bands of the satellite according to the spacecraft provided

        Args:
            spacecraft (str): Spacecraft name

        Returns:
            dataframe: table of bands, which contains the band name and ID in upper case
        """
        if spacecraft in ["LANDSAT_4", "LANDSAT_5"]:
            return pandas.DataFrame(
                [
                    {"NAME": "BLUE", "ID": "1"},
                    {"NAME": "GREEN", "ID": "2"},
                    {"NAME": "RED", "ID": "3"},
                    {"NAME": "NIR", "ID": "4"},
                    {"NAME": "SWIR1", "ID": "5"},
                    {"NAME": "TIRS1", "ID": "6"},
                    {"NAME": "SWIR2", "ID": "7"},
                ]
            )
        if spacecraft in ["LANDSAT_7"]:
            return pandas.DataFrame(
                [
                    {"NAME": "BLUE", "ID": "1"},
                    {"NAME": "GREEN", "ID": "2"},
                    {"NAME": "RED", "ID": "3"},
                    {"NAME": "NIR", "ID": "4"},
                    {"NAME": "SWIR1", "ID": "5"},
                    {"NAME": "TIRS1", "ID": "6_VCID_1"},  # low gain for less satured
                    {"NAME": "SWIR2", "ID": "7"},
                    {"NAME": "PAN", "ID": "8"},
                ]
            )
        if spacecraft in ["LANDSAT_8", "LANDSAT_9"]:
            return pandas.DataFrame(
                [
                    {"NAME": "COASTAL", "ID": "1"},
                    {"NAME": "BLUE", "ID": "2"},
                    {"NAME": "GREEN", "ID": "3"},
                    {"NAME": "RED", "ID": "4"},
                    {"NAME": "NIR", "ID": "5"},
                    {"NAME": "SWIR1", "ID": "6"},
                    {"NAME": "SWIR2", "ID": "7"},
                    {"NAME": "PAN", "ID": "8"},
                    {"NAME": "CIRRUS", "ID": "9"},
                    {"NAME": "TIRS1", "ID": "10"},
                    {"NAME": "TIRS2", "ID": "11"},
                ]
            )
        if spacecraft in ["SENTINEL-2A", "SENTINEL-2B", "SENTINEL-2C"]:
            return pandas.DataFrame(
                [
                    {"NAME": "COASTAL", "ID": "01"},
                    {"NAME": "BLUE", "ID": "02"},
                    {"NAME": "GREEN", "ID": "03"},
                    {"NAME": "RED", "ID": "04"},
                    {"NAME": "VRE1", "ID": "05"},
                    {"NAME": "VRE2", "ID": "06"},
                    {"NAME": "VRE3", "ID": "07"},
                    {"NAME": "WNIR", "ID": "08"},
                    {"NAME": "NIR", "ID": "8A"},
                    {"NAME": "WV", "ID": "09"},
                    {"NAME": "SWIR1", "ID": "11"},
                    {"NAME": "SWIR2", "ID": "12"},
                    {"NAME": "CIRRUS", "ID": "10"},
                ]
            )
        return None

    def get_band_id(self, bandname: str):
        """get band ID by the band name

        Args:
            bandname (str): Band name

        Returns:
            str: Corresponding band ID, which is in string because of the special thermal band case ("6_VCID_1") in Landsat 7
        """
        return self.bands[self.bands["NAME"] == bandname]["ID"].values[0]

    def load_landcover(self):
        """
        Loads the landcover data from a GeoTIFF file if it exists.

        Returns:
            numpy.ndarray: The landcover data as a NumPy array.
        """
        filepath_lc = os.path.join(self.folder, self.name + "_LC.tif")
        if os.path.exists(filepath_lc):
            with rasterio.open(filepath_lc) as src:
                if not src.transform:
                    # Define the geotransform (example values)
                    transform = from_origin(west=0.0, north=0.0, xsize=1.0, ysize=1.0)
                    src.transform = transform
                mask_cover = src.read(1)
            return mask_cover
        else:
            return None

    # %% Methods to be implemented in the subclass
    def load_metadata(self):
        """
        Loads the metadata for the satellite.

        This method must be implemented in the subclass.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Method must be implemented in subclass")

    
        
    def get_saturation(self, band="visible"):
        """
        Get the saturation of the input image based on the specified name.

        Parameters:
            image (numpy.ndarray): The input image.
            name (str): The name of the saturation layer to convert. Default is "visible".

        Returns:
            numpy.ndarray: The converted saturation layer.

        Raises:
            None

        Notes:
            - For the "visible" layer, the image is already processed to the saturated layer for R-G-B bands.
            - For the "green" layer, the image is already processed to the saturated layer for green bands.
        """
        if band == "visible":
            return (self.saturation.get(0) | self.saturation.get(1) | self.saturation.get(2))
        if band == "blue":
            return self.saturation.get(0)
        if band == "green":
            return self.saturation.get(1)
        if band == "red":
            return self.saturation.get(2)
    
    def load_data(self, predictors):
        """
        Loads the data for the satellite.

        Args:
            predictors (list): A list of predictor variables.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.

        Returns:
            None
        """
        raise NotImplementedError("Method must be implemented in subclass")
    
    def clean_saturation(self):
        """Clean the saturation layer and free memory."""
        del self.saturation  # Explicitly delete the object
        self.saturation = None  # Reset the attribute

    def clean_data(self):
        """Clean the datacube and free memory."""
        del self.data
        del self.saturation
        self.data = None
        self.saturation = None

class Landsat(Satellite):
    """Class of Landsat imagery

    Returns:
        object: Object of landsat
    """

    @property
    def tile(self):
        """tile of the image

        Returns:
            tuple: tile of the image
        """
        return self.name.split("_")[2]

    def construct_metadata(self):
        """
        Constructs metadata from the input file.

        Reads the input file, removes trailing newlines, and reconstructs the metadata
        which will be used in further processing.

        Returns:
            None
        """
        self.metafile = os.path.join(self.folder, self.name + "_MTL.txt")
        with open(self.metafile) as f:
            # read through the input file, removing trailing newlines
            lines = [l.strip() for l in f.readlines()]
        # reconstruct the metadata which will be used in the further processing
        meta_groups = [(i,l) for i,l in enumerate(lines) if "GROUP" in l]
        self.metadata: dict = {}
        for i in range(len(meta_groups)-1):
            group = meta_groups[i]
            group_end = meta_groups[i+1]
            group_name = group[1].split("=")[1].strip()
            group_end_name = group_end[1].split("=")[1].strip()
            # if the group is found
            if (group[1] == "GROUP = "+group_name) and (group_end[1] == "END_GROUP = "+group_end_name):
                self.metadata[group_name] = {}
                for j in range(group[0]+1, group_end[0]): # id
                    item_name = lines[j].split("=")[0].strip()
                    item_value = lines[j].split("=")[1].strip()
                    self.metadata[group_name][item_name] = item_value.replace('"', "")
                    # self.metadata[group_name].append({item_name:item_value})
        
    def load_metadata(self):
        """load metadata of the imagery and update the related attributes of the class"""
        self.metafile = os.path.join(self.folder, self.name + "_MTL.json")
        if os.path.exists(self.metafile):
            self.metadata = pandas.read_json(self.metafile)[
                "LANDSAT_METADATA_FILE"
            ]  # only the .json can be loaded into dataframe, reduce the root layer
        else:
            self.construct_metadata()
        self.spacecraft = (self.metadata["IMAGE_ATTRIBUTES"]["SPACECRAFT_ID"]).upper()
        self.sensor = (self.metadata["IMAGE_ATTRIBUTES"]["SENSOR_ID"]).upper()
        self.sun_azimuth = float(
            self.metadata["IMAGE_ATTRIBUTES"]["SUN_AZIMUTH"]
        )  # convert to number in decimal degree
        self.sun_elevation = float(
            self.metadata["IMAGE_ATTRIBUTES"]["SUN_ELEVATION"]
        )  # convert to number in decimal degree
        self.lat_north = np.maximum(
            float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_UL_LAT_PRODUCT"]),
            float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_UR_LAT_PRODUCT"]),
        )
        self.lat_south = np.minimum(
            float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_LL_LAT_PRODUCT"]),
            float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_LR_LAT_PRODUCT"]),
        )
        # add lat_center and lon_center
        self.lat_center = (float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_LL_LAT_PRODUCT"]) + float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_LR_LAT_PRODUCT"]) +
                           float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_UR_LAT_PRODUCT"]) + float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_UL_LAT_PRODUCT"])) / 4
        self.lon_center = (float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_LL_LON_PRODUCT"]) + float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_LR_LON_PRODUCT"]) +
                           float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_UR_LON_PRODUCT"]) + float(self.metadata["PROJECTION_ATTRIBUTES"]["CORNER_UL_LON_PRODUCT"])) / 4
        # build a dataframe for band id and band filename
        self.bands = self.get_spacecraft_bands(
            self.spacecraft
        )  # only the saltellite band remained
        
        # in case of the thermal band is not available for sensor: OLI
        # we need to remove the thermal band from the band list
        if self.sensor == "OLI":
            self.bands = self.bands[self.bands["NAME"] != "TIRS1"]
            self.bands = self.bands[self.bands["NAME"] != "TIRS2"]

    def read_band(self, band, level="TOA", sza=None, maskupdate=True, saturation = False, profile=False):
        """read the raster by the band

        Args:
            band (str): Band name
            level (str, optional): Level of the data. Defaults to 'TOA'.
            sza (2d array, optional): Solar Zenith Angle in radian. Defaults to None.
            maskupdate (bool, optional): Update the observation mask according to the band loading or not. Defaults to True.
            saturation (bool, optional): Return the saturated layer of this band. Defaults to False.
            profile (bool, optional): Update profile of the geoinformation of the imagery. Defaults to False.

        Returns:
            2d array: image of the band
        """
        if C.MSG_FULL:
            print(f">>> loading {band.lower()} in {level.lower()}")
        # specify the band information
        band = band.upper()  # uppercase
        # band_id = self.bands[self.bands['NAME'] == band]['ID'].values[0]
        band_id = self.get_band_id(band)
        path_band = os.path.join(
            self.folder,
            self.metadata["LEVEL1_PROCESSING_RECORD"][f"FILE_NAME_BAND_{band_id}"],
        )

        # read the raw band
        with rasterio.open(path_band) as src:
            image = src.read(1)  # only one band
        if saturation:
            satu = image == np.iinfo(image.dtype).max # 255 for Landsat 4-7, and 65535 for Landsat 8, according to the data's type
        else:
            satu = None
        # update LANDSAT object before converting the pixel value
        # update the extent of the observation when new band is touched
        # in this way, we do not need to load all the bands once time
        if maskupdate:
            if self.obsmask is None:  # initlize the mask of observation extent
                self.obsmask = (
                    image != src.nodata
                )  # should be 0 when we test the Landsat Collection 2.
            else:  # update by overlapping it with new band's extent
                self.obsmask = np.logical_and(
                    self.obsmask, image != src.nodata
                )  # should be 0 when we test the Landsat Collection 2.

        # append the profile of the geotiff, which will be used as the reference to auxiliary data warping and mask saving.
        if profile:
            self.profile = src.meta.copy()
            self.profilefull = src  # save the full profile for the further use

        ## Converting
        image = image.astype(np.float32)
        if level == "TOA":  # Conversion to TOA Reflectance
            REFLECTANCE_MULT = float(
                self.metadata["LEVEL1_RADIOMETRIC_RESCALING"][
                    f"REFLECTANCE_MULT_BAND_{band_id}"
                ]
            )
            REFLECTANCE_ADD = float(
                self.metadata["LEVEL1_RADIOMETRIC_RESCALING"][
                    f"REFLECTANCE_ADD_BAND_{band_id}"
                ]
            )
            image = image * REFLECTANCE_MULT + REFLECTANCE_ADD
            if sza is None:
                sza = self.read_angle(
                    angle="SOLAR_ZENITH", unit="RADIAN"
                )  # when we do not speficfy the angle, we can read it
            image = image / np.cos(sza)  # with a correction for the sun angle
        elif level == "BT":  # Conversion to Top of Atmosphere Brightness Temperature
            # convert to TOA radiance
            RADIANCE_MULT = float(
                self.metadata["LEVEL1_RADIOMETRIC_RESCALING"][
                    f"RADIANCE_MULT_BAND_{band_id}"
                ]
            )
            RADIANCE_ADD = float(
                self.metadata["LEVEL1_RADIOMETRIC_RESCALING"][
                    f"RADIANCE_ADD_BAND_{band_id}"
                ]
            )
            image = image * RADIANCE_MULT + RADIANCE_ADD
            # convert to BT
            K1_CONSTANT = float(
                self.metadata["LEVEL1_THERMAL_CONSTANTS"][f"K1_CONSTANT_BAND_{band_id}"]
            )
            K2_CONSTANT = float(
                self.metadata["LEVEL1_THERMAL_CONSTANTS"][f"K2_CONSTANT_BAND_{band_id}"]
            )
            image = (
                image + C.EPS
            )  #  + np.finfo(float).eps is to avoid divided by 0, of which pixels are filled, and at last fmask will filter them out
            image = K2_CONSTANT / np.log(K1_CONSTANT / image + 1)  # np.log is ln
            # convert from Kelvin to Celcius
            image = image - 273.15
        return image, satu  # return final result

    def read_angle(self, angle, unit="RADIAN", profile=False):
        """read angle bands

        Args:
            angle (str): SENSOR_AZIMUTH, SENSOR_ZENITH, SOLAR_AZIMUTH, SOLAR_ZENITH
            unit (str, optional): radian and degree. Defaults to "RADIAN".

        Returns:
            float: Angle band
        """
        if C.MSG_FULL:
            print(f">>> loading {angle.lower()} in {unit.lower()}")
        angle = angle.upper()  # uppercase
        unit = unit.upper()  # uppercase
        path_angle = os.path.join(
            self.folder,
            self.metadata["LEVEL1_PROCESSING_RECORD"][
                f"FILE_NAME_ANGLE_{angle}_BAND_4"
            ],
        )
        # read the raw band
        with rasterio.open(path_angle) as src:
            image = src.read(1)  # only one band with int16
            # append the profile of the geotiff, which will be used as the reference to auxiliary data warping and mask saving.
            if profile:
                self.profile = src.meta.copy()
                self.profilefull = src  # save the full profile for the further use

        image = image.astype(np.float32) / 100  # rescaling to 0 to 360 degrees
        # convert to radian
        if unit == "RADIAN":
            image = np.deg2rad(image)
        return image

    def read_datacube(self, predictors, maskupdate=True):
        """read all bands defined by predictors

        Args:
            predictors (list): List of band name
            maskupdate (bool, optional): Control to update the observation mask or not. Defaults to True.

        Returns:
            list and 3d array: the list of band name and the datacube
        """
        sza = self.read_angle(angle="SOLAR_ZENITH", unit="RADIAN", profile = True) # update the profile of the geotiff imagery
        # init the bands with 3d dimensions
        bands = np.zeros(
            (len(predictors), self.profile["height"], self.profile["width"]),
            dtype=np.float32,
        )
        # Spectral bands
        predictor = "coastal"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], _ = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate
            )
        predictor = "blue"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], satu_blue = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate, saturation = True
            )
        # saturation for visible bands after the calculation of whiteness
        statu = BitLayer(satu_blue.shape)
        statu.append(satu_blue) # set bit 0 for blue
        del satu_blue

        predictor = "green"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], satu_green = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate, saturation = True
            )
        statu.append(satu_green) # set bit 1 for green
        del satu_green
        
        predictor = "red"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], satu_red  = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate, saturation = True
            )
        statu.append(satu_red) # set bit 2 for red
        del satu_red
    
        predictor = "nir"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], _  = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate
            )
        predictor = "swir1"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], _ = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate
            )
        predictor = "swir2"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], _ = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate
            )
        predictor = "tirs1"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], _ = self.read_band(
                band=predictor, level="BT", maskupdate=maskupdate
            )
        predictor = "tirs2"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], _ = self.read_band(
                band=predictor, level="BT", maskupdate=maskupdate
            )
        predictor = "cirrus"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :], _ = self.read_band(
                band=predictor, level="TOA", sza=sza, maskupdate=maskupdate
            )

        # spectral index
        predictor = "hot"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = hot(
                bands[predictors.index("blue"), :, :],
                bands[predictors.index("red"), :, :],
            )
        predictor = "whiteness"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = whiteness(
                bands[predictors.index("blue"), :, :],
                bands[predictors.index("green"), :, :],
                bands[predictors.index("red"), :, :],
                (statu.get(0) | statu.get(1) | statu.get(2)), # visible bands saturation
            )

        predictor = "ndvi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = ndvi(
                bands[predictors.index("red"), :, :],
                bands[predictors.index("nir"), :, :],
            )
        predictor = "ndsi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = ndsi(
                bands[predictors.index("green"), :, :],
                bands[predictors.index("swir1"), :, :],
            )
        predictor = "ndbi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = ndbi(
                bands[predictors.index("nir"), :, :],
                bands[predictors.index("swir1"), :, :],
            )
        # spatial index
        predictor = "sfdi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = sfdi(
                bands[predictors.index("blue"), :, :],
                bands[predictors.index("swir1"), :, :],
                self.obsmask,
            )
        predictor = "var_nir"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = variation(
                bands[predictors.index("nir"), :, :]
            )
        # surface data
        predictor = "dem"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = gen_dem(self.profile).astype(
                np.float32
            )
        predictor = "swo"
        if predictor in predictors:
            # check the gswo is available for this imagery or not
            if (self.lat_north <= C.NORTH_LAT_GSWO) and (
                self.lat_south >= C.SOUTH_LAT_GSWO
            ):
                bands[predictors.index(predictor), :, :] = gen_gswo(
                    self.profile
                ).astype(np.float32)
            # default value will be zero still, which will not used in the water mask

        # Convert all pixels out of the observation mask to zero
        # according to self.obsmask, change pixel values out of extent to be zero
        if self.obsmask is not None:
            bands[:, ~self.obsmask] = 0  # This flattens mask, not recommended for high-D arrays

        # Fix the pixels in error
        # check any nan values in the bands
        if np.any(np.isnan(bands)):
            bands = np.nan_to_num(bands, nan=C.EPS, posinf=C.EPS, neginf=C.EPS)
        # return the band name and data
        return Data(bands, predictors), statu

    def load_data(self, predictors):
        """
        Reads the dataset for the satellite.

        Args:
            predictors (list): A list of predictors to be read.

        Returns:
            tuple: A tuple containing the datacube and the saturation values.

        """

        self.data, self.saturation = self.read_datacube(predictors)


    def __init__(self, folder):
        # base class init
        super().__init__(folder)
        # load metadata
        self.load_metadata()
        # update the spatial resolution
        self.resolution = 30


class Sentinel2(Satellite):
    """Class of Sentinel imagery"""

    @property
    def tile(self):
        """tile of the image

        Returns:
            tuple: tile of the image
        """
        return self.name.split("_")[5]

    def load_metadata(self):
        # Passing the path of the xml document to enable theparsing process
        self.metafile = os.path.join(self.folder, "MTD_MSIL1C.xml")

        # Passing the path of the xml document to enable theparsing process
        filename_mtd = "MTD_MSIL1C.xml"
        fname = open(os.path.join(self.folder, filename_mtd), "r", encoding="utf-8")
        metadata = ET.parse(fname).getroot()
        fname.close()
        # spacecraft name
        self.spacecraft = metadata.find(".//SPACECRAFT_NAME").text.upper()
        self.sensor = 'MSI' # MSI is the sensor name for Sentinel-2
        # build a dataframe for band id and band filename for Sentinel2
        self.bands = self.get_spacecraft_bands(self.spacecraft)
        band_filename_list = [r.text for r in metadata.findall(".//IMAGE_FILE")]
        # exclude the TCI band, which means True Color Image, that will not be used in the analysis
        band_filename_list = [r for r in band_filename_list if r[-3:] != "TCI"]
        band_id_list = [r[-2:] for r in band_filename_list if r[-3:] != "TCI"]
        # offset and quantification value
        radio_add_offset = [float(r.text) for r in metadata.findall(".//RADIO_ADD_OFFSET")]
        # in case of the offset is not provided, we will use 0, which is for baseline 4.0 and after
        if len(radio_add_offset) == 0:
            radio_add_offset = [0] * len(band_filename_list)

        # make the value as same to the length of the band list
        # same length with the same value
        quantification_value = [float(metadata.find(".//QUANTIFICATION_VALUE").text)] * len(band_filename_list)
        s2_image_files = pandas.DataFrame(
                    {"FILE": band_filename_list,
                    "ID": band_id_list,
                    "RADIO_ADD_OFFSET": radio_add_offset,
                    "QUANTIFICATION_VALUE": quantification_value}
                )
        self.bands = self.bands.merge(s2_image_files, on=["ID"], how="left")

        # get saturation indicator
        SPECIAL_VALUE_TEXT = metadata.findall(".//SPECIAL_VALUE_TEXT")
        SPECIAL_VALUE_INDEX = metadata.findall(".//SPECIAL_VALUE_INDEX")
        self.nodata_index = float([SPECIAL_VALUE_INDEX[itxt] for itxt, txt in enumerate(SPECIAL_VALUE_TEXT) if txt.text == "NODATA"][0].text)
        self.saturation_index = float([SPECIAL_VALUE_INDEX[itxt] for itxt, txt in enumerate(SPECIAL_VALUE_TEXT) if txt.text == "SATURATED"][0].text)

        # get angle metedata
        filename_tl = "MTD_TL.xml"
        folder_tl = Path(
            os.path.join(
                self.folder, self.bands[self.bands["ID"] == "8A"]["FILE"].values[0]
            )
        ).parent.parent
        fname = open(os.path.join(folder_tl, filename_tl), "r", encoding="utf-8")
        metadata = ET.parse(fname)
        fname.close()
        self.sun_elevation = 90 - float(metadata.find(".//ZENITH_ANGLE").text)
        self.sun_azimuth = float(metadata.find(".//AZIMUTH_ANGLE").text)
        self.metaangle = metadata

        # follow the band 8a, which present at 20 meters
        with rasterio.open(
            os.path.join(
                self.folder, self.bands[self.bands["ID"] == "8A"]["FILE"].values[0]
            )
            + ".jp2"
        ) as b8a:
            self.profile = b8a.meta.copy()
            self.profilefull = b8a

        left, bottom, right, top = self.profilefull.bounds
        # x-y to lat-long

        transformer = pyproj.Transformer.from_crs(self.profile["crs"], "epsg:4326")

        # transformer = pyproj.Transformer.from_crs("epsg:{}".format(self.profilefull.crs.to_epsg()), "epsg:4326")

        # transformer.transform(left, top) # x y
        # transformer.transform(right, top)
        # transformer.transform(left, bottom)
        # transformer.transform(right, bottom)

        # pylint: disable= unpacking-non-sequence
        lats_four, lons_four = transformer.transform(
            [left, right, left, right], [top, top, bottom, bottom]
        )  # x y

        # self.lat_range = [np.max(lats_four[0]), np.min(lats_four[0])]  # north, south
        self.lat_north = np.max(lats_four)
        self.lat_south = np.min(lats_four)
        self.lat_center = np.mean(lats_four)
        self.lon_center = np.mean(lons_four)
    def read_band(
        self, band, level="TOA", maskupdate=True, profile=False
    ):  # sza=None is added for the compatibility with the Landsat
        """Reads a specific band from the satellite image.

        Args:
            band (str): The name of the band to read.
            level (str, optional): The level of the image data. Defaults to "TOA".
            maskupdate (bool, optional): Whether to update the observation mask. Defaults to True.
            profile (bool, optional): Whether to append the profile of the geotiff. Defaults to False.

        Returns:
            numpy.ndarray: The image data of the specified band.

        Raises:
            IndexError: If the specified band is not found in the bands metadata.

        Note:
            - The No Data value is represented as 0 in the L1C product. See https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-1c/product-formatting
            - The image data is returned as a numpy array, scaled to the range [0, 1] by dividing by 10000 if the level is "TOA".
        """

        if C.MSG_FULL:
            print(f">>> loading {band.lower()} in {level.lower()}")
        # The No Data value is represented as 0 in the L1C product. see https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/level-1c/product-formatting
        src_nodata = self.nodata_index
        # specify the band information
        band = band.upper()  # uppercase
        band_file = self.bands[self.bands["NAME"] == band]["FILE"].values[0]
        res_trgt = self.profilefull.res[
            0
        ]  # resolution of processing images, such as 20 m
        with rasterio.open(os.path.join(self.folder, band_file) + ".jp2") as src:
            res_band = src.res[0]
            if res_trgt == res_band:  # band's resolution == our target
                image = src.read(1)  # only one band, and orginal band
            elif res_trgt > res_band:  # band's resolution < our target's
                image = src.read(
                    1, out_shape=self.profilefull.shape, resampling=Resampling.average
                )  # only one band, and average is used to aggregate to larger resolution
            elif res_trgt < res_band:  # band's resolution < our target's
                image = src.read(
                    1, out_shape=self.profilefull.shape, resampling=Resampling.nearest
                )  # only one band, and nearest is used to resample to smaller resolution
        if maskupdate:
            # self.obsmask = np.logical_and(self.obsmask, image != src.nodata) # should be 0 when we test the Landsat Collection 2.
            if self.obsmask is None:  # initlize the mask of observation extent
                self.obsmask = (
                    image != src_nodata
                )  # should be 0 when we test the Landsat Collection 2.
            else:  # update by overlapping it with new band's extent
                self.obsmask = np.logical_and(
                    self.obsmask, image != src_nodata
                )  # should be 0 when we test the Landsat Collection 2.

        # append the profile of the geotiff, which will be used as the reference to auxiliary data warping and mask saving.
        if profile:
            self.profile = src.meta.copy()
            self.profilefull = src  # save the full profile for the further use

        # return final result
        if level == "TOA":
            return self.convert_dn_to_toa(band, image)
        else:
            return image

    def convert_dn_to_toa(self, band, data):
        """
        Converts the Digital Number (DN) values to Top of Atmosphere (TOA) reflectance values for a given band.

        Args:
            band (str): The name of the band to convert.
            data (numpy.ndarray): The DN values to convert.

        Returns:
            numpy.ndarray: The TOA reflectance values for the specified band.

        Raises:
            IndexError: If the specified band is not found in the bands metadata.
        """
        # Eq: L1C_DN = TOA * QUANTIFICATION_VALUE - RADIO_ADD_OFFSET
        quanti = self.bands[self.bands["NAME"] == band.upper()]["QUANTIFICATION_VALUE"].values[0]
        offset = self.bands[self.bands["NAME"] == band.upper()]["RADIO_ADD_OFFSET"].values[0]
        return (data + offset) / quanti
   
    def set_view_angles(self, bandid = 7):
        """
        Sets the view angles for the satellite.

        This method parses the XML file containing the satellite metadata and extracts the viewing incidence angles.
        The extracted angles are then stored in the `_sensor_zenith` and `_sensor_azimuth` attributes of the satellite object.
        Credits: https://github.com/brazil-data-cube/s2-angs/blob/master/s2angs/s2_angs.py
        Note:
        - The method assumes that the XML file has already been loaded into the `metaangle` attribute of the satellite object.
        - The method uses bandId 7 for the viewing incidence angles, as it is commonly used in vegetation applications.
        - The method resizes the angle matrices to 22x22, as the angle bands pixels represent 5000 meters and the Sentinel-2 images cover an area of 109800x109800 meters.
        
        Returns:
        None
        """
        numband = 13
        sensor_zenith_values = np.empty((numband,23,23)) * np.nan #initiates matrix
        sensor_azimuth_values = np.empty((numband,23,23)) * np.nan

        # Parse the XML file
        root = self.metaangle.getroot()

        # Find the angles
        for child in root:
            if child.tag[-14:] == 'Geometric_Info':
                geoinfo = child

        for segment in geoinfo:
            if segment.tag == 'Tile_Angles':
                angles = segment

        for angle in angles:
            if angle.tag == 'Viewing_Incidence_Angles_Grids':
                bandId = int(angle.attrib['bandId'])
                for bset in angle:
                    if bset.tag == 'Zenith':
                        zenith = bset
                    if bset.tag == 'Azimuth':
                        azimuth = bset
                for field in zenith:
                    if field.tag == 'Values_List':
                        zvallist = field
                for field in azimuth:
                    if field.tag == 'Values_List':
                        avallist = field
                for rindex in range(len(zvallist)):
                    zvalrow = zvallist[rindex]
                    avalrow = avallist[rindex]
                    zvalues = zvalrow.text.split(' ')
                    avalues = avalrow.text.split(' ')
                    values = list(zip(zvalues, avalues )) #row of values
                    for cindex in range(len(values)):
                        if (values[cindex][0] != 'NaN' and values[cindex][1] != 'NaN'):
                            zen = float(values[cindex][0])
                            az = float(values[cindex][1])
                            sensor_zenith_values[bandId, rindex,cindex] = zen
                            sensor_azimuth_values[bandId, rindex,cindex] = az
        # In the next two lines, 7 is adopted as bandId since for our application we opted to not generate the angle bands for each of the spectral bands. Here we adopted bandId 7 due to its use in vegetation applications
        # Also on the next two lines, we are using 22x22 matrices since the angle bands pixels represent 5000 meters. Sentinel-2 images area 109800x109800 meters. The 23x23 5000m matrix is equivalent to 11500x11500m. Based on that we opted to not use the last column and row. More information can be found on STEP ESA forum: https://forum.step.esa.int/t/generate-view-angles-from-metadata-sentinel-2/5598
        # for cloud detection as well, we followed Landsat's angle which is based NIR band as well  
        # fill nan by the nearest value by row and then by column
        self._sensor_zenith = resize(fill_nan_wise(sensor_zenith_values[bandid]),(22,22)) # nan value will cause all pixels as nan
        self._sensor_azimuth = resize(fill_nan_wise(sensor_azimuth_values[bandid]),(22,22))

    def read_angle(self, angle, unit="RADIAN"):
        """
        Reads the angle data for a given angle type and converts it to the specified unit.

        Args:
            angle (str): The type of angle to read. Valid options are "SENSOR_AZIMUTH" and "SENSOR_ZENITH".
            unit (str, optional): The unit to convert the angle to. Defaults to "RADIAN".

        Returns:
            numpy.ndarray: The angle data as a numpy array.

        Raises:
            ValueError: If an invalid angle type is provided.

        """

        if C.MSG_FULL:
            print(f">>> loading {angle.lower()} in {unit.lower()}")

        if angle.upper() == "SENSOR_AZIMUTH":
            if self._sensor_azimuth is None:
                self.set_view_angles() # set up the 22 pixels by 22 pixels matrix for the angles for both azimuth and zenith
            image = resize(self._sensor_azimuth, (self.profile["height"], self.profile["width"]))
        elif angle.upper() == "SENSOR_ZENITH":
            if self._sensor_zenith is None:
                self.set_view_angles()
            image = resize(self._sensor_zenith, (self.profile["height"], self.profile["width"]))
        else:
            raise ValueError("Invalid angle type. Valid options are 'SENSOR_AZIMUTH' and 'SENSOR_ZENITH'.")

        # convert to radian
        if unit == "RADIAN":
            image = np.deg2rad(image)
        return image

    def read_datacube(self, predictors, maskupdate=True):
        """
        Reads the data cube for the given predictors.

        Args:
            predictors (list): List of predictor names.
            maskupdate (bool, optional): Whether to update the mask. Defaults to True.

        Returns:
            tuple: A tuple containing the data cube and the saturation mask.
                - data (Data): The data cube containing the bands and their values.
                - statu (numpy.ndarray): The saturation mask indicating the saturation status of the visible bands.
        """
        # init the bands with 3d dimensions
        bands = np.zeros(
            (len(predictors), self.profile["height"], self.profile["width"]),
            dtype=np.float32,
        )
        # Spectral bands
        predictor = "coastal"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "blue"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "green"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "red"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "vre1"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "vre2"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "vre3"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )

        predictor = "wnir"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "nir"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "wv"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "swir1"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "swir2"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )
        predictor = "cirrus"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = self.read_band(
                band=predictor, level="TOA", maskupdate=maskupdate
            )

        # spectral index
        predictor = "hot"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = hot(
                bands[predictors.index("blue"), :, :],
                bands[predictors.index("red"), :, :],
            )

        # saturation for visible bands after the calculation of whiteness
        statu = BitLayer(bands.shape[1:])
        statu.append(bands[predictors.index("blue"), :, :] == self.convert_dn_to_toa("blue", self.saturation_index))
        statu.append(bands[predictors.index("green"), :, :] == self.convert_dn_to_toa("green", self.saturation_index))
        statu.append(bands[predictors.index("red"), :, :] == self.convert_dn_to_toa("red", self.saturation_index))

        predictor = "whiteness"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = whiteness(
                bands[predictors.index("blue"), :, :],
                bands[predictors.index("green"), :, :],
                bands[predictors.index("red"), :, :],
                (statu.get(0) | statu.get(1) | statu.get(2)), # visible bands saturation
            )

        predictor = "ndvi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = ndvi(
                bands[predictors.index("red"), :, :],
                bands[predictors.index("nir"), :, :],
            )
        predictor = "ndsi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = ndsi(
                bands[predictors.index("green"), :, :],
                bands[predictors.index("swir1"), :, :],
            )
        predictor = "ndbi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = ndbi(
                bands[predictors.index("nir"), :, :],
                bands[predictors.index("swir1"), :, :],
            )
        # spatial index
        predictor = "cdi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = cdi(
                bands[predictors.index("vre3"), :, :],
                bands[predictors.index("wnir"), :, :],
                bands[predictors.index("nir"), :, :],
            )
        predictor = "sfdi"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = sfdi(
                bands[predictors.index("blue"), :, :],
                bands[predictors.index("swir1"), :, :],
                self.obsmask,
            )
        predictor = "var_nir"
        if predictor in predictors:
            if C.MSG_FULL:
                print(f">>> calculating {predictor}")
            bands[predictors.index(predictor), :, :] = variation(
                bands[predictors.index("nir"), :, :]
            )
        # surface data
        predictor = "dem"
        if predictor in predictors:
            bands[predictors.index(predictor), :, :] = gen_dem(self.profile).astype(
                np.float32
            )
        predictor = "swo"
        if predictor in predictors:
            # check the gswo is available for this imagery or not
            if (self.lat_north <= C.NORTH_LAT_GSWO) and (
                self.lat_south >= C.SOUTH_LAT_GSWO
            ):
                bands[predictors.index(predictor), :, :] = gen_gswo(
                    self.profile
                ).astype(np.float32)
            # default value will be zero still, which will not used in the water mask
        
        # Convert all pixels out of the observation mask to zero
        # according to self.obsmask, change pixel values out of extent to be zero
        if self.obsmask is not None:
            bands[:, ~self.obsmask] = 0  # This flattens mask, not recommended for high-D arrays

        # Fix the pixels in error
        if np.any(np.isnan(bands)):
            bands = np.nan_to_num(bands, nan=C.EPS, posinf=C.EPS, neginf=C.EPS)

        # return the band name and data
        return Data(bands, predictors), statu

    def load_data(self, predictors):
        """
        Reads the dataset for the satellite.

        Args:
            predictors (list): A list of predictors to be read.

        Returns:
            tuple: A tuple containing the datacube and the saturation values.

        """
        # read saturate qa for visible bands
        self.data, self.saturation = self.read_datacube(predictors)

    def __init__(self, folder):
        # base class init
        super().__init__(folder)
        # load metadata
        self.load_metadata()
        # update the spatial resolution
        self.resolution = 20
