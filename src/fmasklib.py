# pylint: disable=line-too-long
import os
import sys
from pathlib import Path
from typing import Union
import pandas as pd
from satellite import Landsat, Sentinel2
from phylib import flood_fill_shadow
import predictor as P
import utils
from unetlib import UNet
from lightgbmlib import Dataset as PixeDataset
from lightgbmlib import LightGBM
import constant as C
from phylib import Physical, segment_cloud_objects
from bitlib import BitLayer
import numpy as np
from skimage.filters import threshold_otsu
np.seterr(invalid='ignore') # ignore the invalid errors

class Fmask(object):
    """Fmask class
    """

    # Cloud detection algorithm
    algorithm = "interaction"  # the algorithm for cloud masking, including "physical", "randomforest", "unet", "interaction"

    # Image object can be hold either a Landsat or Sentinel2 object
    image: Union[Landsat, Sentinel2] = None

    # The physical cloud detection model
    physical: Physical = None
    
    # The lightgbm cloud detection model
    lightgbm_cloud: LightGBM = None
    
    # The unet cloud detection model
    unet_cloud: UNet = None  # the unet model for cloud masking

    # The pixel dataset for training the pixel-based model, like lightgbm
    # database_pixel: PixeDataset = None
    pixelbase: PixeDataset = None  # the dataset for the lightgbm model
    patchbase = None  # the dataset for the UNet model
    database_patch = None  # the dataset for UNet model

    path = None
    dir_patch = None  # directory of patches for training the unet model
    dir_pixel = None  # directory of patches for training the lightgbm

    # predictor_full = None  # full predictors that are provided by the module
    # spatial resolution that we processing the image, like 30m for Landsat and 20m for Sentinel-2
    resolution = 30

    # the radius of erosion, unit: pixels, for the postprocessing of cloud objects, see Fmask 4.0 paper for details
    erosion_radius = 0  # the radius of erosion, unit: pixels
    dilation_radius_unet = 0 # not used for now, since we have addressed the ommission issues at image boundarys by shifting the image chips

    # the buffer size of cloud, shadow, and snow in pixels (its resolution is the same as the image's resolution)
    buffer_cloud = 0
    buffer_shadow = 5 # the buffer size of shadow in pixels, which is larger than the original size, 3 by 3 pixels, since the larger dilation size is able better to fill the holes caused by the projection of clouds (to match shadow)
    buffer_snow = 0

    # The classes of the cloud and non-cloud, and filled pixels for the machine learning model
    # the pixel value will rely on the index of the defined classes
    cloud_model_classes = ["noncloud", "cloud", "filled"]

    base_machine_learning = ["unet"]  # the base machine learning model for cloud masking, such as 'unet', 'lightgbm', 'lightgbm_unet'
    tune_machine_learning = "lightgbm"  # the machine learning model for tuning the cloud masking, such as 'unet', 'lightgbm'
    tune_strategy = "transfer"  # 'transfer' or 'new'
    pixel_erosion_radius = 0  # the radius of erosion for water for excluding potential false water pixels for training lightgbm, unit: pixels
    max_iteration = (
        1  # maximum iteration numbers between fmask and machine learning model
    )
    disagree_rate = 0.25  # the rate of disagreement between two consective iteration by machine learning. It will not be used if the max_iteration is 1
    seed_levels = [0, 0] # percentile of selecting non-cloud seeds and cloud seeds
 
    physical_rules_dynamic = True  # able to change rules during the iterations

    # the valid pixel values in reference data
    valid_class_labels = [
        C.LABEL_CLEAR,
        C.LABEL_WATER,
        C.LABEL_LAND,
        C.LABEL_SNOW,
        C.LABEL_SHADOW,
        C.LABEL_CLOUD,
        C.LABEL_FILL,
    ]

    # sets of displaying figures to show the progress of the cloud masking
    show_figure = False

    # Masks
    cloud: BitLayer = None  # the cloud masks, which can store multiple cloud masks by bit layers
    cloud_region = None  # the cloud region list, will be used in shadow masking
    cloud_object = None  # the cloud object mask, will be used in shadow masking
    shadow = None  # the shadow mask
    probability = None  # the cloud probability layer

    @property
    def full_predictor(self) -> list:
        """
        Returns a list of predictors based on the selected algorithm.

        Returns:
            list: A list of predictors to be used in the model.
        """
        # select the predictors according to the algorithm given
        if self.algorithm == "physical":
            predictors = self.physical.predictors.copy()
        elif self.algorithm == "lightgbm":
            predictors = self.lightgbm_cloud.predictors.copy()
            predictors = predictors + self.physical.predictors # no matter what, we need to use the physical predictors to create variables to match shadows
        elif self.algorithm == "unet":
            predictors = self.unet_cloud.predictors.copy()
            predictors = predictors + self.physical.predictors
        elif self.algorithm == "interaction":
            predictors = self.physical.predictors.copy()
            if ("unet" in self.base_machine_learning) | (self.tune_machine_learning == "unet"):
                predictors = predictors + self.unet_cloud.predictors
            if ("lightgbm" in self.base_machine_learning) | (self.tune_machine_learning == "lightgbm"):
                predictors = predictors + self.lightgbm_cloud.predictors
        predictors = list(set(predictors)) # unique the predictors
        # sort the predictors according to the full predictors of the spacecraft
        image_spacecraft = self.image.spacecraft.upper()
        if image_spacecraft in ["LANDSAT_8", "LANDSAT_9"]:
            predictor_full = P.l8_predictor_full.copy()
        elif image_spacecraft in ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7"]:
            predictor_full = P.l7_predictor_full.copy()
        elif image_spacecraft in ["SENTINEL-2A", "SENTINEL-2B", "SENTINEL-2C"]:
            predictor_full = P.s2_predictor_full.copy()
        else:
            raise ValueError(f"Unsupported spacecraft: {image_spacecraft}")
        predictors.sort(key=lambda x: predictor_full.index(x) if x in predictor_full else float('inf'))
        return predictors # unique and same sorted the predictors

    @property
    def ensemble_mask(self):
        """
        Generates an ensemble mask based on different classification results.

        Returns:
            numpy.ndarray: The ensemble mask with labeled regions for water, snow, shadow, cloud, and fill.
        """
        if self.cloud is None:
            return None
        else:
            mask = np.zeros(self.image.obsmask.shape, dtype="uint8")
            if self.physical is not None:
                mask[self.physical.water] = C.LABEL_WATER
                mask[self.physical.snow] = C.LABEL_SNOW
            if self.shadow is not None:
                if self.buffer_shadow > 0:
                    mask[utils.dilate(self.shadow, radius=self.buffer_shadow)] = C.LABEL_SHADOW
                else:
                    mask[self.shadow] = C.LABEL_SHADOW
            # the cloud mask must exist
            if self.buffer_cloud > 0:
                mask[utils.dilate(self.cloud.last, radius=self.buffer_cloud)] = C.LABEL_CLOUD
            else:
                mask[self.cloud.last] = C.LABEL_CLOUD
            mask[self.image.filled] = C.LABEL_FILL
            # convert to uint8
            mask = mask.astype("uint8")
            return mask

    @property
    def cloud_percentage(self):
        """
        Returns the percentage of cloud coverage in the image.

        Returns:
            float: The percentage of cloud coverage.
        """
        return np.count_nonzero(np.bitwise_and(self.cloud.last, self.image.obsmask)) / np.count_nonzero(self.image.obsmask)

    def set_base_machine_learning(self, models: str) -> None:
        """
        Sets the base machine learning model for the cloud masking algorithm.

        Args:
            models (str): The base machine learning model to set, with each model separated by an underscore.
        """
        self.base_machine_learning = models.split("_")

    def set_tune_machine_learning(self, model: str) -> None:
        """
        Sets the tune machine learning model for the cloud masking algorithm.

        Args:
            models (str): The base machine learning model to set, with each model separated by an underscore.
        """
        self.tune_machine_learning = model

    def get_patch_data_index(self, predictors, datalayers=None) -> list:
        """
        Get the index of the predictors that are used in the patch dataset.

        Returns:
            list: The index of the predictors.
        """
        # see create_train_data_patch.py for the predictors used in the patch dataset generation process
        if datalayers is None:
            datalayers = P.l8_predictor_cloud_cnn
        return [
            i for i, pre in enumerate(datalayers) if pre in predictors
        ]

    # %% Methods
    def init_modules(self, loadmodel = True) -> None:
        """
        Initialize and optimize the cloud models based on the spacecraft type.

        This method initializes and configures the cloud models based on the spacecraft type.
        It sets the appropriate parameters and values for each model.

        Returns:
            None
        """
        # initialize the cloud models without initialization according to the spacecraft
        spacecraft = self.image.spacecraft.upper()
        if spacecraft in ["LANDSAT_8", "LANDSAT_9"]:
            self.physical = Physical(
                predictors=P.l8_predictor_cloud_phy.copy(), woc=0.3, threshold=0.175, overlap=0.0
            )
            if(self.algorithm == "lightgbm") or (self.algorithm == "interaction" and (self.tune_machine_learning == "lightgbm" or "lightgbm" in self.base_machine_learning)):
                if loadmodel:
                    path_model = os.path.join(self.dir_package, "model", "lightgbm_cloud_l8.pk")
                else:
                    path_model = None
                self.lightgbm_cloud = LightGBM(
                    classes=["noncloud", "cloud"],
                    num_leaves=30,
                    min_data_in_leaf=700,
                    tune_update_rate=0.03,
                    predictors=P.l8_predictor_cloud_pixel.copy(),
                    path=path_model,
                )
            if(self.algorithm == "unet") or (self.algorithm == "interaction" and (self.tune_machine_learning == "unet" or "unet" in self.base_machine_learning)):
                if loadmodel:
                    path_model = os.path.join(self.dir_package, "model", "unet_cloud_l8.pt") # the unet model for Landsat 4-9
                else:
                    path_model = None
                self.unet_cloud = UNet(
                    classes=["noncloud", "cloud", "filled"],
                    predictors=P.l8_predictor_cloud_cnn.copy(),
                    learn_rate=1e-3,
                    epoch=40,
                    patch_size=256,
                    patch_stride_train=224,
                    patch_stride_classify=224,
                    tune_epoch=5,
                    path=path_model,
                )
            self.resolution = 30  # spatial resolution that we processing the image
            self.erosion_radius = int(90/self.resolution)  # the radius of erosion, unit: pixels
            self.dilation_radius_unet = 0 # unit: pixels

        elif spacecraft in ["LANDSAT_4", "LANDSAT_5", "LANDSAT_7"]:
            self.physical = Physical(
                predictors=P.l7_predictor_cloud_phy.copy(), woc=0.0, threshold=0.1, overlap=0.0
            )
            if self.algorithm == "lightgbm" or (self.algorithm == "interaction" and (self.tune_machine_learning == "lightgbm" or "lightgbm" in self.base_machine_learning)):
                if loadmodel:
                    path_model = os.path.join(self.dir_package, "model", "lightgbm_cloud_l7.pk")
                else:
                    path_model = None
                self.lightgbm_cloud = LightGBM(
                    classes=["noncloud", "cloud"],
                    num_leaves=30,
                    min_data_in_leaf=700,
                    tune_update_rate=0.03,
                    predictors=P.l7_predictor_cloud_pixel.copy(),
                    path=path_model,
                )
            if self.algorithm == "unet" or (self.algorithm == "interaction" and (self.tune_machine_learning == "unet" or "unet" in self.base_machine_learning)):
                if loadmodel:
                    path_model = os.path.join(self.dir_package, "model", "unet_cloud_l7.pt") # the unet model for Landsat 4-9
                else:
                    path_model = None
                self.unet_cloud = UNet(
                    classes=["noncloud", "cloud", "filled"],
                    predictors=P.l7_predictor_cloud_cnn.copy(),
                    learn_rate=1e-3,
                    epoch=40,
                    patch_size=256,
                    patch_stride_train=224,
                    patch_stride_classify=224,
                    tune_epoch=5,
                    path=path_model,
                )
            self.resolution = 30  # spatial resolution that we processing the image
            self.erosion_radius = int(150/self.resolution)  # the radius of erosion, unit: pixels
            self.dilation_radius_unet = 0 # unit: pixels

        elif spacecraft in ["SENTINEL-2A", "SENTINEL-2B", "SENTINEL-2C"]:
            self.physical = Physical(
                predictors=P.s2_predictor_cloud_phy.copy(), woc=0.5, threshold=0.2, overlap=0.0
            )
            if self.algorithm == "lightgbm" or (self.algorithm == "interaction" and (self.tune_machine_learning == "lightgbm" or "lightgbm" in self.base_machine_learning)):
                if loadmodel:
                    path_model = os.path.join(self.dir_package, "model", "lightgbm_cloud_s2.pk")
                else:
                    path_model = None
                self.lightgbm_cloud = LightGBM(
                    classes=["noncloud", "cloud"],
                    num_leaves=40,
                    min_data_in_leaf=500,
                    tune_update_rate=0.03,
                    predictors=P.s2_predictor_cloud_pixel.copy(),
                    path=path_model,
                )
            if self.algorithm == "unet" or (self.algorithm == "interaction" and (self.tune_machine_learning == "unet" or "unet" in self.base_machine_learning)):
                if loadmodel:
                    path_model = os.path.join(self.dir_package, "model", "unet_cloud_s2.pt")
                else:
                    path_model = None
                self.unet_cloud = UNet(
                    classes=["noncloud", "cloud", "filled"],
                    predictors=P.s2_predictor_cloud_cnn.copy(),
                    learn_rate=1e-3,
                    epoch=40,
                    patch_size=256,
                    patch_stride_train=224,
                    patch_stride_classify=224,
                    tune_epoch=5,
                    path=path_model,
                )

            self.resolution = 20  # spatial resolution that we processing the image
            self.erosion_radius = int(90/self.resolution)   # the radius of erosion, unit: pixels
            self.dilation_radius_unet = 0 # unit: pixels

    def init_pixelbase(
        self,
        directory=None,
        datasets=None,
        classes=None,
        sampling_methods=None,
        number=20000,
        exclude=None,
    ) -> None:
        """initialize the dataset for training the pixel-based model

        Args:
            directory (str): The directory to save the dataset.
            datasets (list): The datasets to use for collecting training data.
            classes (list): The classes of the dataset.
            sampling_methods (list): The sampling methods of the dataset.
            number (int): The number of samples to collect.
            exclude (str): Image will be excluded from the training data.

        """
        # locate the pixel datasets
        spacecraft = self.image.spacecraft.upper()
        if spacecraft.startswith("L"):
            if directory is None:
                directory = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataPixel1/Landsat8"
                datasets = ["L8BIOME", "L8SPARCS", "L895CLOUD"]
        elif spacecraft.startswith("S"):
            if directory is None:
                directory = "/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/TrainingDataPixel1/Sentinel2"
                datasets = ["S2ALCD", "S2WHUCDPLUS", "S2FMASK" , "S2IRIS",  "S2CLOUDSEN12PLUS"]
        if classes is None:
            classes = self.lightgbm_cloud.classes
        if sampling_methods is None:
            sampling_methods = ["stratified", "stratified"]
        # init the pixelbase
        self.pixelbase = PixeDataset(
            directory,
            datasets,
            classes,
            sampling_methods,
            number=number,
            exclude=exclude,
        )
        # forward the dataset to the model
        self.lightgbm_cloud.sample = self.pixelbase

    def load_image(self) -> None:
        """Load image according to the configuration and forward image to the models

        This method loads the image using the specific bands that are known.
        It then forwards the dataset to the models for further processing.

        Args:
            None

        Returns:
            None
        """
        # load image with the specific bands that we know
        self.image.load_data(self.full_predictor)

        # forward dataset to the models, without tripling the dataset
        if self.physical is not None:
            self.physical.image = self.image
        if self.lightgbm_cloud is not None:
            self.lightgbm_cloud.image = self.image
        if self.unet_cloud is not None:
            self.unet_cloud.image = self.image

    def generate_train_data_pixel(
        self, dataset, number, destination=None
    ) -> pd.DataFrame:
        """Collects training data for pixels.

        This method collects training data for pixels based on the specified dataset and number of samples.
        It saves the collected training data as a CSV file if a path is provided.

        Args:
            dataset (str): The dataset to use for collecting training data.
            path (str): The path to save the collected training data. (default: None)
            number (int): The number of samples to collect.

        Returns:
            pf_sample (Dataframe): The collected training data as a pandas DataFrame.
        """
        landcover = self.image.load_landcover()
        reference = utils.read_reference_mask(
            self.image.folder, dataset=dataset, shape=self.image.shape
        )

        # check the reference mask's classes
        labels_in = np.unique(reference)
        print(f">>> unique values in the reference mask {labels_in}")
        labels_invalid = labels_in[
            [lab not in self.valid_class_labels for lab in labels_in]
        ]
        if len(labels_invalid) > 0:
            print(
                f">>> the reference mask is not in the valid classes {labels_invalid}"
            )
            sys.exit(0)  # exit the program if error occurs

        # start to collect the samples
        pf_sample = utils.collect_sample_pixel(
            self.image.data.data,
            self.image.data.bands,
            reference,
            landcover=landcover,
            number=number,
        )
        print(f">>> {len(pf_sample):012d} samples have been collected")
        # create the directory if it does not exist
        if destination is not None:
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            pf_sample.to_csv(destination)
            print(f">>> saved to {destination}")
            
        return pf_sample

    def generate_train_data_patch(
        self, dataset, path, dformat="tif", shift=True
    ) -> None:
        """
        Generate training data patches.

        Args:
            dataset (str): The dataset to generate patches for.
            path (str): The path to save the generated patches.
            shift (bool, optional): Whether to append patches to the end of the existing dataset. Defaults to True.
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        reference = utils.read_reference_mask(
            self.image.folder, dataset=dataset, shape=self.image.shape
        )

        # check the reference mask's classes
        labels_in = np.unique(reference)
        print(f">>> unique values in the reference mask {labels_in}")
        labels_invalid = labels_in[
            [lab not in self.valid_class_labels for lab in labels_in]
        ]
        if len(labels_invalid) > 0:
            print(
                f">>> the reference mask is not in the valid classes {labels_invalid}"
            )
            sys.exit(0)  # exit the program if error occurs

        # start to collect the samples
        utils.collect_sample_patch(
            dataset,
            self.image.name,
            self.image.profile,
            self.image.profilefull,
            self.image.data,
            reference,
            self.image.obsmask,
            path,
            size=self.unet_cloud.patch_size,  # the size of the patch
            stride=self.unet_cloud.patch_stride_train,  # the stride of the patch
            shift=shift,
            dformat=dformat,
        )

    def mask_cloud_pcp(self):
        """
        Masks the pixel cloud probability (pcp) and assigns the result to the `cloud` attribute.
        Also assigns the absolute clear probability (abs_clear) to the `shadow` attribute.
        """
        # init cloud layer
        if self.physical.activated is None:
            self.physical.init_cloud_probability()
        self.cloud = BitLayer(self.image.shape)
        self.cloud.append(self.physical.pcp)

    def mask_shadow(self, postprocess, min_area = 3, buffer2connect = 0, potential = "flood", topo="SCS", thermal_adjust = True, threshold = 0.15):
        """
        Masks the shadow in the image based on the given parameters.
        Parameters:
        postprocess (str): Indicates whether post-processing should be applied.
        min_area (int): The minimum area of the shadow to be considered.
        potential (str, optional): The method to be used for potential shadow detection. Defaults to "flood".
        thermal_include (bool, optional): Indicates whether thermal information should be included. Defaults to True, only for Landsat.
        Returns:
        None
        Notes:
        - If the cloud percentage is greater than or equal to 90%, the rest of the pixels are directly identified as cloud shadow.
        - If the algorithm is "interaction" or "physical" and the physical model is activated, cloud shadow matching is skipped due to high cloud coverage.
        - Otherwise, a cloud object is created and shadow geometry is masked based on the given potential method.
            """
        # if self.cloud_percentage >= 0.9 or (
        #     self.algorithm in {"interaction", "physical"} and (not self.physical.activated)
        # ):
        if self.cloud_percentage >= 0.9:
            # Skip cloud shadow matching when cloud coverage is too high
            if C.MSG_FULL:
                print(">>> skipping cloud shadow matching due to high cloud coverage.")
            self.mask_shadow_rest()
        elif self.cloud_percentage > 0: # only when there are some clouds, we need to match the shadows
            self.create_cloud_object(postprocess=postprocess, min_area=min_area, buffer2connect=buffer2connect)
            self.mask_shadow_geometry(potential=potential, topo=topo, thermal_adjust=thermal_adjust, threshold = threshold)

    def mask_shadow_rest(self):
        """Masks the cloud shadow on the left side of the cloud."""
        self.shadow = ~self.cloud.last  # Bitwise negation for masks

    def mask_cloud_physical(self):
        """mask clouds by the default physical rules"""
        # init cloud layer
        self.cloud = BitLayer(self.image.shape)

        if self.physical.activated is None:
            self.physical.init_cloud_probability()

        if self.physical.activated:
            _options = self.physical.options.copy()  # save the original options
            self.physical.options = [
                True,
                True,
                True,
            ]  # make sure all the physical rules are activated at default
            
            cloud_ph = self.physical.cloud
            # also get the extremely cold clouds when thermal band is available
            cold_cloud = self.physical.cold_cloud
            if cold_cloud is not None:
                cloud_ph[cold_cloud] = 1
            # self.cloud.append(self.physical.cloud)
            self.cloud.append(cloud_ph)
            # show cloud probabilities
            if self.show_figure:
                # show the cloud probabilities
                if self.physical.options[0]:
                    utils.show_cloud_probability(
                        self.physical.prob_variation,
                        self.physical.image.filled,
                        "Spectral variation probability",
                    )
                if self.physical.options[1]:
                    utils.show_cloud_probability(
                        self.physical.prob_temperature,
                        self.physical.image.filled,
                        "Temperature probability",
                    )
                if self.physical.options[2]:
                    utils.show_cloud_probability(
                        self.physical.prob_cirrus,
                        self.physical.image.filled,
                        "Cirrus probability",
                    )
                # make title with the options at the end (TTT)
                utils.show_cloud_probability(
                    self.physical.prob_cloud,
                    self.physical.image.filled,
                    f"Cloud probability ({str(self.physical.options[0])[0]}{str(self.physical.options[1])[0]}{str(self.physical.options[2])[0]})",
                )
            self.physical.options = _options.copy()  # recover the original options
            del _options  # empty the variable that will not be used anymore
        else:  # get the pcp
            self.cloud.append(self.physical.pcp)
        # show cloud mask
        if self.show_figure:
            cloud_mask = self.cloud.last.astype("uint8")
            cloud_mask[self.image.filled] = self.cloud_model_classes.index("filled")
            utils.show_cloud_mask(
                cloud_mask, self.cloud_model_classes, "Physical rules"
            )

    # post processing and generate cloud objects
    def create_cloud_object(self, min_area = 3, postprocess = "none", buffer2connect = 0):
        """
        Erodes the false positive cloud pixels.
        Returns:
            None
        """
        cloud_layer = self.cloud.last
        # exclude the filled pixels
        cloud_layer[self.image.filled] = 0
        if (postprocess == "none"):
            # no postprocessing
            [cloud_objects, cloud_regions] = segment_cloud_objects(cloud_layer, min_area=min_area)
            if min_area > 0: # only when min_area > 0, we need to remove the small cloud objects, and then we need to update the cloud layer recorded
                self.cloud.append(cloud_objects >0)
        elif (postprocess == "morphology"):
            # only when the physical model is activated
            # if not self.physical.activated:
            #    return
            # morphology-based, follow Qiu et al., 2019 RSE
            if C.MSG_FULL:
                print(">>> postprocessing with morphology-based elimination")
            # get the potential false positive cloud pixels
            pfpl = self.mask_potential_bright_surface()
            # erode the false positive cloud pixels
            pixels_eroded = utils.erode(cloud_layer, radius = self.erosion_radius)
            pfpl = (~pixels_eroded) & pfpl & (~self.physical.water) & self.image.obsmask # indicate the eroded cloud pixels over land
            del pixels_eroded

            # segment the cloud pixels into objects, and if all the pixels of cloud are over the eroded layer, then remove the cloud object
            # remove the small cloud objects less than 3 pixels
            [cloud_objects, cloud_regions] = segment_cloud_objects(cloud_layer, min_area=min_area, exclude=pfpl, exclude_method = 'all')
            del pfpl

            # remove the small cloud objects 
            if self.image.data.exist("cdi"):
                # Pre-calculate the mask for `_cdi > -0.5
                cdi_mask = self.image.data.get("cdi") > -0.5
                false_small_cloud = np.zeros_like(cloud_objects, dtype=bool) # initialize the cloud_objects as 0
                valid_indices = []
                for icloud, cld_obj in enumerate(cloud_regions):
                    if (cld_obj.area < 10000) and (np.all(cdi_mask[cld_obj.coords[:, 0], cld_obj.coords[:, 1]])): # Check if all cdi values are > -0.5
                        false_small_cloud[cld_obj.coords[:, 0], cld_obj.coords[:, 1]] = True
                    else:
                        valid_indices.append(icloud)
                cloud_objects[false_small_cloud] = 0
                del false_small_cloud, cdi_mask
                cloud_regions = [cloud_regions[i] for i in valid_indices]
                del valid_indices
            # after postprocessing
            self.cloud.append(cloud_objects >0)
        elif (postprocess == "unet"):
            # only when the physical model is activated
            # if not self.physical.activated:
            #    return
            if C.MSG_FULL:
                print(">>> postprocessing with UNet-based elimination")
            # Use dilated version only if needed
            unet_cloud = self.cloud.first
            unet_cloud = utils.dilate(unet_cloud, radius=self.dilation_radius_unet) if self.dilation_radius_unet > 0 else unet_cloud
            [cloud_objects, cloud_regions] = segment_cloud_objects(cloud_layer, min_area=min_area, exclude=(unet_cloud==0), exclude_method = 'all')
            del unet_cloud
            # after postprocessing
            self.cloud.append(cloud_objects >0)
        elif (postprocess == "morphology_unet"):
            if C.MSG_FULL:
                print(">>> postprocessing with morphology&unet-based elimination")
            # Use dilated version only if needed
            unet_cloud = self.cloud.first
            unet_cloud = utils.dilate(unet_cloud, radius=self.dilation_radius_unet) if self.dilation_radius_unet > 0 else unet_cloud
             # get the potential false positive cloud pixels
            pfpl = self.mask_potential_bright_surface()
            # erode the false positive cloud pixels
            pixels_eroded = utils.erode(cloud_layer, radius = self.erosion_radius)
            pfpl = (~pixels_eroded) & pfpl & (~self.physical.water) & self.image.obsmask # indicate the eroded cloud pixels over land
            del pixels_eroded
            pfpl = pfpl | (unet_cloud==0) # exclude the unet cloud pixels
            del unet_cloud
            # segment the cloud pixels into objects, and if all the pixels of cloud are over the eroded layer, then remove the cloud object
            # remove the small cloud objects less than 3 pixels
            [cloud_objects, cloud_regions] = segment_cloud_objects(cloud_layer, min_area=min_area, exclude=pfpl, exclude_method = 'all')
            del pfpl

            # remove the small cloud objects 
            if self.image.data.exist("cdi"):
                # Pre-calculate the mask for `_cdi > -0.5
                cdi_mask = self.image.data.get("cdi") > -0.5
                false_small_cloud = np.zeros_like(cloud_objects, dtype=bool) # initialize the cloud_objects as 0
                valid_indices = []
                for icloud, cld_obj in enumerate(cloud_regions):
                    if (cld_obj.area < 10000) and (np.all(cdi_mask[cld_obj.coords[:, 0], cld_obj.coords[:, 1]])): # Check if all cdi values are > -0.5
                        false_small_cloud[cld_obj.coords[:, 0], cld_obj.coords[:, 1]] = True
                    else:
                        valid_indices.append(icloud)
                cloud_objects[false_small_cloud] = 0
                del false_small_cloud, cdi_mask
                cloud_regions = [cloud_regions[i] for i in valid_indices]
                del valid_indices
            # after postprocessing
            self.cloud.append(cloud_objects >0)
        
        # only do this after making the postprocessing for cloud objects, such as the mininum size of small cloud objects
        if buffer2connect > 0:
            [cloud_objects, cloud_regions] = segment_cloud_objects(cloud_objects >0, buffer2connect=buffer2connect)
        # assign the cloud objects and regions 
        self.cloud_object = cloud_objects
        self.cloud_region = cloud_regions
        # update the cloud mask after postprocessing

    def mask_potential_bright_surface(self):
        """
        Masks the potential bright surface pixels in the image.
        Returns:
            numpy.ndarray: The mask indicating the potential bright surface pixels.
        """
        
        # over urban by use the ndbi
        ndbi = self.image.data.get("ndbi")
        ndbi = utils.enhance_line(ndbi)
        # urban pixels
        pfpl = (ndbi > 0) & (ndbi > self.image.data.get("ndvi")) & (~self.physical.water)
        
        if np.any(pfpl): # only when the potential false positive cloud pixels are available
            # ostu threshold to exclude cloud over the layer if the thermal band is available
            if self.image.data.exist("tirs1"):
                # exclude the extremely cold pixels over the urban pixels
                cold_t = threshold_otsu(self.image.data.get("tirs1")[pfpl])  # Otsu's method
                pfpl[self.image.data.get("tirs1") < cold_t] = 0
            # exclude the confident cloud pixels by cdi
            if self.image.data.exist("cdi"):
                pfpl[self.image.data.get("cdi") < -0.8] = 0  # Follow David, 2018 RSE for Sentinel-2
        
        # Add potential snow/ice pixels in mountain areas
        if self.image.data.exist("slope"):
            _slope = self.image.data.get("slope")
        else:
            _slope = utils.calculate_slope(self.image.data.get("dem"))
        # potential snow/ice pixels in mountain areas
        pfpl = pfpl | ((_slope > 20) & self.physical.snow)

        # Buffer urban pixels with a 500 window to connect the potential false positive cloud pixels into one layer
        radius_pixels = int(250 / self.image.resolution)  # 1 km = 33 Landsat pixels, 500m = 17, 200m = 7
        pfpl = utils.dilate(pfpl, radius=radius_pixels)
        
        # add the snow pixels in normal regions with no dilation in mountain areas
        pfpl = pfpl | self.physical.snow

        return pfpl
            
    def mask_cloud_interaction(self, outcome="classified"):
        """Mask clouds by the interaction of the physical model and machine learning model

        Args:
            outcome (str, optional): Can be "classified" or "physical". Defaults to "classified".
        """
        
        ## Special case: when the update rate is zero or tune_epoch is zero, we do not need to update the model, just use the base model as the base
        # test only, which can be removed in loclean version
        if outcome == "classified":
            if self.tune_machine_learning == "unet" and self.unet_cloud.tune_epoch == 0:
                self.mask_cloud_unet()
                return
            elif self.tune_machine_learning == "lightgbm" and self.lightgbm_cloud.tune_update_rate == 0:
                self.mask_cloud_lightgbm()
                return

        # init cloud layer in bit layers
        self.cloud = BitLayer(self.image.shape)

        #%% init cloud probabilities
        if self.physical.activated is None:
            self.physical.init_cloud_probability()
        # # check the physical model activated or not
        # if not self.physical.activated:
        #     print(
        #         ">>> physical model has not been initialized due to inadquate absolute clear-sky pixels"
        #     )
        #     self.cloud.append(self.physical.pcp)
        #     return

        # display cloud probabilities from the physical rules
        if self.show_figure:
            # show water layer
            utils.show_simple_mask(self.physical.water, "Water")
            
            # show the cloud probabilities
            utils.show_cloud_probability(
                self.physical.prob_variation,
                self.physical.image.filled,
                "Spectral variation",
            )
            utils.show_cloud_probability(
                self.physical.prob_temperature,
                self.physical.image.filled,
                "Temperature/HOT",
            )
            utils.show_cloud_probability(
                self.physical.prob_cirrus,
                self.physical.image.filled,
                "Cirrus",
            )
            # to display the full cloud probability
            self.physical.options = [
                True,
                True,
                True,
            ]
            
            utils.show_cloud_probability(
                self.physical.prob_cloud,
                self.physical.image.filled,
                f"Cloud probability ({str(self.physical.options[0])[0]}{str(self.physical.options[1])[0]}{str(self.physical.options[2])[0]})"
            )

        #%% create initial cloud mask by the machine learning model
        # start to process normal imagery
        # define the class label of the cloud and non-cloud, and filled pixels for the machine learning model
        label_cloud     = self.cloud_model_classes.index("cloud")
        label_noncloud  = self.cloud_model_classes.index("noncloud")
        label_filled    = self.cloud_model_classes.index("filled")

        # load the pretrained unet model if it will be used
        if C.MSG_FULL:
            print(f">>> loading {self.base_machine_learning} as base machine learning model")
            print(f">>> loading {self.tune_machine_learning} as tune machine learning model")
    
        # load the pretrained machine learning models required accordingly
        if (("unet" in self.base_machine_learning) or (self.tune_machine_learning == "unet")) and (not self.unet_cloud.activated):
            self.unet_cloud.load_model()
        if (("lightgbm" in self.base_machine_learning) or (self.tune_machine_learning == "lightgbm")) and (not self.lightgbm_cloud.activated):
            self.lightgbm_cloud.load_model()
            
        
        # masking initilized clouds
        # if C.MSG_FULL:
        #    print(f">>> initilizing cloud mask by {self.base_machine_learning}")
        # get the init mask created by the machine learning model
        # single classifier is used as base machine learning model
        if "unet" in self.base_machine_learning:
            cloud_ml, _ = self.unet_cloud.classify(probability="none")
            cloud_ml[self.image.filled] = label_filled  # exclude the filled pixels by the real extent masking
        elif "lightgbm" in self.base_machine_learning:
            cloud_ml, _, subsampling_mask = self.lightgbm_cloud.classify(probability="none", base = True)
            # exclude the filled pixels based on the subsampling mask, which is used to speed up the classification of lightgbm
            cloud_ml[~subsampling_mask] = label_filled

        
        # # code of testing random forest to take the classification of physical rules as the input, based on the unet cloud mask
        # from sklearn.ensemble import RandomForestClassifier
        # from sklearn.datasets import make_classification
        # # train a random forest classifier
        # # randomly select 20000 pixels from the cloud_ml that are not filled
        # rf_idx = np.where((cloud_ml != label_filled) & (self.image.obsmask))
        # selected_indices = np.random.choice(len(rf_idx[0]), size=20000, replace=False)
        # rf_idx = (rf_idx[0][selected_indices], rf_idx[1][selected_indices])
        # rf_y = cloud_ml[rf_idx]
        # # get the predictors from the physical model
        # rf_x = np.zeros((len(rf_idx[0]), 3), dtype="float32")
        # rf_x[:,0] = self.physical.prob_variation[rf_idx]
        # rf_x[:,1] = self.physical.prob_temperature[rf_idx]
        # rf_x[:,2] = self.physical.prob_cirrus[rf_idx]
        # # train the random forest classifier
        # rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        # rf_clf.fit(rf_x, rf_y)
        # # predict the cloud mask for all the pixels that are not filled
        # rf_x_all = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype="float32")
        # rf_x_all[:,:,0] = self.physical.prob_variation
        # rf_x_all[:,:,1] = self.physical.prob_temperature
        # rf_x_all[:,:,2] = self.physical.prob_cirrus
        # cloud_rf = cloud_ml.copy()
        # cloud_rf[cloud_ml != label_filled] = rf_clf.predict(rf_x_all[cloud_ml != label_filled].reshape(-1, 3))
        
        # if self.show_figure:
        #     utils.show_cloud_mask(
        #         cloud_rf, self.cloud_model_classes, "Random Forest based on physical probabilities"
        #     )
        
        #%% iteract only when the cloud and non-cloud are both in the cloud_ml        
        count_cloud = np.count_nonzero(cloud_ml == label_cloud)
        count_noncloud = np.count_nonzero(cloud_ml == label_noncloud)
        # when the pixel-based classifiers are used, we can consider the subsampling sizen to speed up the classification
        if ("lightgbm" in self.base_machine_learning):
            # one pixel represent the subsampling_size * subsampling_size pixels after using subsampling to speed up the classification of pixel-based classifiers
            # back to the original number of pixels before subsampling
            ratio_subsampling = self.image.obsnum/np.count_nonzero(cloud_ml != label_filled)
            subsampling_mask = None # only when the first iteration, we do not need to use the subsampling mask because it was based on the random pixels selected for triggering the physical model
            count_cloud = count_cloud*ratio_subsampling
            count_noncloud = count_noncloud*ratio_subsampling
            del ratio_subsampling

         # record the cloud layer
        if (outcome == "classified") or ((count_cloud < self.physical.min_clear) or (count_noncloud < self.physical.min_clear)):
            # when outcome is classified,
            # when either cloud or non-cloud is not represented in the initial cloud_ml, we do not do the interaction, just record the initial cloud_ml
            self.cloud.append(cloud_ml == label_cloud)
        # else: # we do not use this because we only record the physical outcome at the end when the physical model swift is made
        #     self.physical.options = [
        #         True,
        #         True,
        #         True,
        #     ]  # make sure all the physical rules are activated at default
        #     self.cloud.append(
        #         self.physical.cloud
        #     )  # get the default physical cloud mask
           
        if self.show_figure:
            utils.show_cloud_mask(
                cloud_ml, self.cloud_model_classes, "base: " + "".join(self.base_machine_learning)
            )
            # utils.show_cloud_probability(
            #    prob_ml, self.image.filled, f"base: {self.base_machine_learning_string}"
            # )

        # both cloud and non-cloud are represented enough in the cloud_ml
        if (count_cloud >= self.physical.min_clear) & (count_noncloud >= self.physical.min_clear):
            # check the physical model activated or not
            if not self.physical.activated:
                print(
                    ">>> physical model has not been initialized due to inadquate absolute clear-sky pixels"
                )
                # return the PCP pixels, and will do the post-processing for the cloud objects overlaying the UNet cloud layer
                self.cloud.append(self.physical.pcp)
                return
        # if (count_cloud >= 1) & (count_noncloud >= 1):
            # count the labels of cloud and non-cloud
            # self-learning progress
            for i in range(1, self.max_iteration + 1):
                if C.MSG_FULL:
                    print(
                        f">>> adjusting physical rules {i:02d}/{self.max_iteration:02d}"
                    )

                # select the seeds of cloud and non-cloud (which is not used for any further processing, just for information)
                # if self.show_figure:
                #    utils.show_seed_mask(
                #        cloud_ml,
                #        cloud_ml,
                #        self.cloud_model_classes,
                #        "Seed: " + "&".join(self.base_machine_learning),
                #    )

                # physical rules and Control to make the rules combined dynamically
                if (i == 1) or self.physical_rules_dynamic: # i == 1 does not require the physical rules to be adjusted
                    self.physical.set_options() # back to the default options, which means dynamically adjust the rules, on or off
                else:  # only use the physical rules determined by the initilization when we do not need to adjust the rules anymore
                    self.physical.set_options([options[0]], [options[1]], [options[2]])

                # select the cloud probability layer by the physical rules
                (prob_ph, options, thrd) = self.physical.select_cloud_probability(
                    cloud_ml,
                    label_cloud = label_cloud,
                    label_noncloud = label_noncloud,
                    show_figure = self.show_figure
                )
                # adjust the threshold by the threshold shift
                thrd = max(0, thrd - self.physical.threshold_shift) # decrease the threshold to make sure more cloud pixels are included, but not too much
                # mask cloud by the physical rules
                cloud_ph = np.zeros(self.image.shape, dtype="uint8")
                if label_noncloud != 0: # only when it is not zero, we update the mask of noncloud
                    cloud_ph[prob_ph < thrd]= label_noncloud
                # see overlap_cloud_probability, in which we counted this included to mask clouds
                # have to be PCP pixels
                cloud_ph[(prob_ph >= thrd) & self.physical.pcp] = label_cloud
                # exclude extremely cold clouds when thermal band is available
                if self.image.data.exist("tirs1"):
                    cloud_ph[self.physical.cold_cloud] = label_cloud
                # set the mask of non-observed pixels to filled
                cloud_ph[self.image.filled] = label_filled

                if self.show_figure:
                    # make title with the options at the end (TTT)
                    utils.show_cloud_probability(
                        self.physical.prob_cloud,
                        self.physical.image.filled,
                        f"Cloud probability ({str(self.physical.options[0])[0]}{str(self.physical.options[1])[0]}{str(self.physical.options[2])[0]})"
                    )
                    utils.show_cloud_mask(
                        cloud_ph, self.cloud_model_classes, "Physical rules"
                    )

                # when the outcome is physical at the end, we do not run the machine learning classification one more time.
                if (i == self.max_iteration) & (outcome == "physical"):
                    self.cloud.append(cloud_ph == label_cloud)
                    return

                # continue to classify and tune the machine learning model
                if C.MSG_FULL:
                    print(f">>> tunning machine learning model {i:02d}/{self.max_iteration:02d}")
                # tune the unet and return the cloud mask and cloud probability layer by the updated model
                if self.tune_machine_learning == "unet":
                    (cloud_ml_update, _) = self.unet_cloud.tune(cloud_ph)  # as seed layer
                elif (self.tune_machine_learning == "lightgbm"):
                    # Exclude the place where the physical rules often make omission errors of cloud pixels.
                    # Note commision errors are not considered here because post-processing will be made, such as UNet-based postprocess for UPU or morphology-based postprocess for LPL
                    # 1. Non-cloud pixels near the cloud pixels, where the physical rules often make omission errors. In this case, the remaining non-cloud pixels far away from clouds still can serve as the training samples.
                    if self.pixel_erosion_radius > 0:
                        cloud_ph[(cloud_ph!=label_cloud) & utils.dilate(cloud_ph==label_cloud, radius = self.pixel_erosion_radius)] = label_filled

                    # update the training data, and retrain the pixel-based model
                    # update the training data by replacing the samples
                    if self.lightgbm_cloud.tune_update_rate > 0:
                        self.lightgbm_cloud.sample.update(
                            self.image.data.get(self.lightgbm_cloud.predictors),
                            self.lightgbm_cloud.predictors,
                            cloud_ph,
                            label_cloud=label_cloud,
                            label_fill=label_filled,
                            number=int(
                                self.lightgbm_cloud.tune_update_rate
                                * self.lightgbm_cloud.sample.number
                            ),
                            method="replace",
                        )
                        # retrain the pixel-based model only when we have the samples updated
                        self.lightgbm_cloud.train()
                    (cloud_ml_update, _,_) = self.lightgbm_cloud.classify()
                else: # in case of the default, but will not come to this line
                    cloud_ml_update = cloud_ml.copy()

                # exclude the filled pixels by the real extent masking
                cloud_ml_update[self.image.filled] = (
                    label_filled  # set the mask of non-observed pixels to filled
                )

                if self.show_figure:
                    utils.show_cloud_mask(
                        cloud_ml_update, self.cloud_model_classes, self.tune_machine_learning
                    )

                # disagreement between two layers and update the cloud mask
                disagree_ml = (
                    1
                    - np.count_nonzero(
                        (cloud_ml_update == cloud_ml) & self.image.obsmask
                    )
                    / self.image.obsnum
                )
                # update to the new cloud mask and cloud probability layer
                cloud_ml = cloud_ml_update.copy()

                # update the cloud mask by the machine learning model
                if outcome == "classified":
                    # make a buffer to connect the cloud pixels when the subsampling is used
                    # this was designed to speed up the classification using pixel-based classifiers, but after testing, it will harm the classification results, so we do not use it
                    if (self.tune_machine_learning == "lightgbm") and (self.lightgbm_cloud.subsampling_size > 1):
                        self.cloud.append(utils.dilate(cloud_ml == label_cloud, radius=self.lightgbm_cloud.subsampling_size-1))
                    else:
                        self.cloud.append(cloud_ml == label_cloud)
                else:
                    self.cloud.append(cloud_ph == label_cloud)

                # reach to the end iteration
                if (i == self.max_iteration):
                    if C.MSG_FULL:
                        print(
                            ">>> stop iterating at the end"
                        )
                    return

                # stop by the disagreement rate
                if disagree_ml < self.disagree_rate:
                    if C.MSG_FULL:
                        print(
                            f">>> stop iterating with disagreement = {disagree_ml:.2f} less than {self.disagree_rate}"
                        )
                    return

                # stop if the cloud and non cloud seed pixels are not enough to represent 
                count_cloud = np.count_nonzero(cloud_ml == label_cloud)
                count_noncloud = np.count_nonzero(cloud_ml == label_noncloud)
                if (count_cloud < self.physical.min_clear) | (count_noncloud < self.physical.min_clear):
                    if C.MSG_FULL:
                        print(
                            f">>> stop iterating with less representive seed pixels for cloud = {count_cloud} for noncloud = {count_noncloud}"
                        )
                    return


    def mask_cloud_unet(self, probability="cloud") -> None:
        """mask clouds by UNet

        Args:
            probability (str, optional): "cloud": cloud prob. "noncloud": noncloud prob. or "none": not to extract the prob layer. "default": highest score for the classified results. Defaults to "none".
        """
        # init cloud layer
        self.cloud = BitLayer(self.image.shape)

        self.unet_cloud.load_model()
        if self.show_figure: # show the cloud mask and cloud probability figures
            _cloud, prob_ml = self.unet_cloud.classify(probability=probability)
        else: # force to none
            _cloud, _ = self.unet_cloud.classify(probability="none")
        # append cloud layer
        self.cloud.append(
            _cloud == self.unet_cloud.classes.index("cloud")
        )  # make the cloud mask as binary, in which 1 is cloud and 0 is non-cloud
        
        # show the cloud mask and cloud probability figures at the end
        if self.show_figure:
            cloud_mask = self.cloud.last.astype("uint8")
            cloud_mask[self.image.filled] = self.cloud_model_classes.index("filled")
            utils.show_cloud_mask(cloud_mask, self.cloud_model_classes, "UNet")
            utils.show_cloud_probability(
                prob_ml, self.unet_cloud.image.filled, "Cloud Probability"
            )

    def mask_cloud_lightgbm(self, probability="cloud") -> None:
        """
        Masks clouds in the image using LightGBM.
        This method initializes the LightGBM model if it is not already activated,
        creates a cloud mask layer, classifies the cloud probability, and updates
        the cloud mask. Optionally, it can display the cloud mask and cloud probability
        figures.
        Args:
            probability (str, optional): The probability type to use for cloud classification.
                                         Defaults to "cloud".
        Returns:
            None
        """
        
        # init model
        if not self.lightgbm_cloud.activated:
            self.lightgbm_cloud.load_model()

        # init cloud layer
        self.cloud = BitLayer(self.image.shape)
        
        # classify the cloud, and get the cloud probability layer only when we want to show the figure
        if self.show_figure:
            (_cloud, prob_ml, _) = self.lightgbm_cloud.classify(
                probability=probability
            )  # the cloud probability layer, its definition is based on the classes
        else: # just cloud layer returned
            (_cloud, _, _) = self.lightgbm_cloud.classify(
                probability="none" # no need to process the cloud probability layer
            )
        # append to the final cloud layer
        self.cloud.append(
            _cloud == self.lightgbm_cloud.classes.index("cloud")
        )  # make the cloud mask as binary, in which 1 is cloud and 0 is non-cloud

        # check if we need to show the figure
        if self.show_figure:
            cloud_mask = self.cloud.last.astype("uint8")
            cloud_mask[self.image.filled] = self.cloud_model_classes.index("filled")
            utils.show_cloud_mask(cloud_mask, self.cloud_model_classes, "LightGBM")
            utils.show_cloud_probability(
                prob_ml, self.unet_cloud.image.filled, "Cloud Probability"
            )

    def display_predictor(self, band, title=None, percentiles=None) -> None:
        """
        Display the predictor for a given band of the image.
        Parameters:
        band (str): The name of the band to display.
        title (str, optional): The title for the display. Defaults to None.
        percentiles (list of float, optional): The percentiles to use for scaling the display. Defaults to None.
        Returns:
        None
        """
        
        _band = self.image.data.get(band)
        _band[self.image.filled] = np.nan
        # _band = np.interp(_band, np.nanpercentile(_band, percentiles), [0, 1])
        vrange = np.nanpercentile(_band, percentiles)
        utils.show_predictor(_band, self.image.filled, title, vrange= vrange)

    def display_image(
        self, bands=None, title=None, percentiles=None, path=None, min_range = None
    ) -> None:
        """Display a color image composed of specified bands.

        Args:
            bands (list, optional): List of band names to compose the color image. Defaults to ["red", "green", "blue"].
            title (str, optional): Title of the displayed image. Defaults to None.
            percentiles (list, optional): List of percentiles to use for contrast stretching. Defaults to [2, 98].
        """
        if bands is None:
            bands = ["red", "green", "blue"]
        if title is None:
            title = f"Color image ({bands[0]}, {bands[1]}, {bands[2]})"
        if percentiles is None:
            percentiles = [2, 98]
        rgb, r_range, b_range, g_range = utils.composite_rgb(
            self.image.data.get(bands[0]),
            self.image.data.get(bands[1]),
            self.image.data.get(bands[2]),
            self.image.obsmask,
            percentiles=percentiles,
            min_range=min_range, # just for the display of the cirrus band with mininum value = 0.01
        )
        # only when the path is not None, we will save the image
        if path is not None:
            # append the range values on the file path
            fig_name = os.path.basename(path)
            fig_ext = os.path.splitext(fig_name)[1]
            fig_name = os.path.splitext(fig_name)[0]
            # the numbers with 4 decimal points
            r_range = [round(r_range[0], 4), round(r_range[1], 4)]
            g_range = [round(g_range[0], 4), round(g_range[1], 4)]
            b_range = [round(b_range[0], 4), round(b_range[1], 4)]

            # when the bands are same, only one band is used
            if bands[0] == bands[1] == bands[2]:
                # append to filename
                path = os.path.join(
                    os.path.dirname(path),
                    fig_name
                    + f"_{r_range[0]}_{r_range[1]}"
                    + fig_ext
                )
            else:
                # append to filename
                path = os.path.join(
                    os.path.dirname(path),
                    fig_name
                    + f"_R_{r_range[0]}_{r_range[1]}_G_{g_range[0]}_{g_range[1]}_B_{b_range[0]}_{b_range[1]}"
                    + fig_ext
                )
        utils.show_image(rgb, title, path)

    def check_mask_existence(self, endname=None):
        """
        Check if a mask file exists for the given image.
        This method constructs the filename for the mask file based on the image's
        destination, name, and the provided endname (or the algorithm name if 
        endname is not provided). It then checks if a file with that name exists.
        Args:
            endname (str, optional): The suffix to append to the image name to form 
                                     the mask filename. If None, the algorithm name 
                                     is used.
        Returns:
            bool: True if the mask file exists, False otherwise.
        """
        
        if endname is None:
            endname = self.algorithm
        return os.path.isfile(os.path.join(self.image.destination, self.image.name + "_" + endname.upper() + ".tif"))
        
    def save_mask(self, endname=None) -> None:
        """Save the mask to the specified path.

        Args:
            path (str): The path to save the mask.
            mask (ndarray): The mask to save.
            classes (list): The classes of the mask.
            title (str): The title of the mask.
            format (str, optional): The format of the mask. Defaults to "GTiff".
        """
        # get the mask
        emask = self.ensemble_mask
        # update the profile
        profile = self.image.profile.copy()
        # profile["dtype"] = type(emask)  # update the dtype accordingly
        profile["dtype"] = "uint8"
        if endname is None:
            endname = self.algorithm
        # create the directory if it does not exist
        Path(self.image.destination).mkdir(parents=True, exist_ok=True)
        utils.save_raster(
            emask.astype("uint8"),
            profile,
            os.path.join(
                self.image.destination, self.image.name + "_" + endname.upper() + ".tif"
            ),
        )
        if C.MSG_FULL:
            print(f">>> saved fmask layer as geotiff to {self.image.destination}")
        

    def save_model_metadata(self, path, running_time=0.0) -> None:
        """save model's metadata to a CSV file

        Args:
            path (_type_): _description_
            running_time (float, optional): The running time of the algorithm. Defaults to 0.0.
        """
        df_accuracy = pd.DataFrame.from_dict(
            [
                {
                    "image": self.image.name,
                    "model": self.algorithm,
                    "cloud_percentage": self.cloud_percentage,
                    "spectral_variation": self.physical.options[0],
                    "temperature_hot": self.physical.options[1],
                    "cirrus": self.physical.options[2],
                    "threshold": self.physical.threshold,
                    "running_time": running_time,
                }
            ],
            orient="columns",
        )
        # create the directory if it does not exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # print(df_accuracy)
        df_accuracy.to_csv(path)
        if C.MSG_FULL:
            print(f">>> saved metadata as csv file to {path}")
        

    def save_accuracy(self, dataset, path, running_time=0.0, shadow=False):
        """Saves the accuracy metrics of the cloud and shadow masks to a CSV file.

        Parameters:
            dataset (str): The dataset name.
            endname (str, optional): The suffix to append to the output file name. Defaults to None.
            running_time (float, optional): The running time of the algorithm. Defaults to 0.0.
            shadow (bool, optional): Flag indicating whether to include shadow accuracy metrics. Defaults to False.

        Returns:
            None
        """
        # Import only when needed
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        # Function code goes here
        emask = self.ensemble_mask
        # read the manual mask
        mmask = utils.read_reference_mask(
            self.image.folder, dataset=dataset, shape=emask.shape
        )
        # when we do not get the accruacy of shadow layer
        if not shadow:
            mmask[mmask == C.LABEL_SHADOW] = (
                C.LABEL_CLEAR
            )  # we consider shadow as clear
            emask[emask == C.LABEL_SHADOW] = (
                C.LABEL_CLEAR
            )  # we consider shadow as clear
        emask[emask == C.LABEL_LAND] = (
            C.LABEL_CLEAR
        )  # update as to clear label to validate
        emask[emask == C.LABEL_WATER] = (
            C.LABEL_CLEAR
        )  # update as to clear label to validate
        emask[emask == C.LABEL_SNOW] = (
            C.LABEL_CLEAR
        )  # update as to clear label to validate
        # same extent between the manual mask and the ensemble mask
        mmask[emask == C.LABEL_FILL] = C.LABEL_FILL  # same extent with the manual mask
        emask[mmask == C.LABEL_FILL] = C.LABEL_FILL  # same extent with the manual mask

        mmask = mmask[mmask != C.LABEL_FILL]
        emask = emask[emask != C.LABEL_FILL]

        # Cloud, Shadow, and Clear
        csc_overall = accuracy_score(mmask, emask)

        cloud_precision, shadow_precision = precision_score(
            mmask,
            emask,
            labels=[C.LABEL_CLOUD, C.LABEL_SHADOW],
            average=None,
            zero_division=1.0,
        )
        cloud_recall, shadow_recall = recall_score(
            mmask,
            emask,
            labels=[C.LABEL_CLOUD, C.LABEL_SHADOW],
            average=None,
            zero_division=1.0,
        )

        # Cloud, and Non-cloud (cloud shadow and clear)
        cn_overall = accuracy_score(mmask == C.LABEL_CLOUD, emask == C.LABEL_CLOUD)

        # Cloud percentage
        cloud_percentage_pred = np.count_nonzero(emask == C.LABEL_CLOUD) / len(emask)
        cloud_percentage_true = np.count_nonzero(mmask == C.LABEL_CLOUD) / len(mmask)

        # Cloud shadow percentage
        shadow_percentage_pred = np.count_nonzero(emask == C.LABEL_SHADOW) / len(emask)
        shadow_percentage_true = np.count_nonzero(mmask == C.LABEL_SHADOW) / len(mmask)

        # Number of observaiont pixels
        num_obs_pred = len(emask)
        num_obs_true = len(mmask)

        df_accuracy = pd.DataFrame.from_dict(
            [
                {
                    "image": self.image.name,
                    "cloud_percentage_pred": cloud_percentage_pred,
                    "cloud_percentage_true": cloud_percentage_true,
                    "shadow_percentage_pred": shadow_percentage_pred,
                    "shadow_percentage_true": shadow_percentage_true,
                    "cn_overall": cn_overall,
                    "csc_overall": csc_overall,
                    "cloud_precision": cloud_precision,
                    "cloud_recall": cloud_recall,
                    "shadow_precision": shadow_precision,
                    "shadow_recall": shadow_recall,
                    "num_obs_pred": num_obs_pred,
                    "num_obs_true": num_obs_true,
                    "running_time": running_time,
                }
            ],
            orient="columns",
        )
        # create the directory if it does not exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        print(df_accuracy)
        df_accuracy.to_csv(path)

    def mask_shadow_flood(self, topo='SCS', threshold = 0.15):
        """
        Mask cloud shadows based on the flood method.
        Args:
            topo (str, optional): Topographic correction method. Defaults to 'SCS'. None means does not apply topographic correction.
        Returns:
            nd.array(bool): shadow mask by fill flooded
        """
        if topo is None:
            return flood_fill_shadow(
                self.image.data.get("nir"),
                self.image.data.get("swir1"),
                self.physical.abs_clear_land,
                self.image.obsmask,
                threshold = threshold,
            )
        else:
            slope = utils.gen_slope(self.image.profile)
            aspect = utils.gen_aspect(self.image.profile)
            if self.show_figure:
                # to show the orginal and topo-corrected bands
                percentiles = [2, 98]
                _band_nir = self.image.data.get("nir")
                _band_swir = self.image.data.get("swir1")
                _band_nir_cor, _band_swir_cor= utils.topo_correct_scs(_band_nir, _band_swir, self.image.sun_elevation, self.image.sun_azimuth, slope, aspect)
                
                _band_nir[self.image.filled] = np.nan
                vrange = np.nanpercentile(_band_nir, percentiles)
                utils.show_predictor(_band_nir, self.image.filled, 'Orginal NIR', vrange= vrange)
                _band_nir_cor[self.image.filled] = np.nan
                vrange = np.nanpercentile(_band_nir_cor, percentiles)
                utils.show_predictor(_band_nir_cor, self.image.filled, 'SCS-corrected NIR', vrange= vrange)
                
                _band_swir[self.image.filled] = np.nan
                vrange = np.nanpercentile(_band_swir, percentiles)
                utils.show_predictor(_band_swir, self.image.filled, 'Orginal SWIR1', vrange= vrange)
                _band_swir_cor[self.image.filled] = np.nan
                vrange = np.nanpercentile(_band_swir_cor, percentiles)
                utils.show_predictor(_band_swir_cor, self.image.filled, 'SCS-corrected SWIR1', vrange= vrange)

            if topo == "SCS":
                # check if any clear_land pixels is available
                _abs_land_pixels = self.physical.abs_clear_land
                if np.any(_abs_land_pixels): # the background will be computed by percentiles in the flood_fill_shadow function if we have clear_land pixels
                    nir_background=None 
                    swir1_background=None
                else: # just can be zero when there are no clear land pixels, to mask shadow , we need a small value
                    nir_background = 0
                    swir1_background = 0
                # correct the topo for the NIR and SWIR1 bands at the same time
                nir_cor, swir_cor = utils.topo_correct_scs(self.image.data.get("nir"), self.image.data.get("swir1"),  self.image.sun_elevation, self.image.sun_azimuth, slope, aspect)
                return flood_fill_shadow(nir_cor, swir_cor, _abs_land_pixels, self.image.obsmask,
                                         threshold = threshold,
                                         nir_background = nir_background,
                                         swir1_background = swir1_background)

    def mask_shadow_geometry(self, potential="flood", topo = 'SCS', thermal_adjust = True, threshold = 0.15):
        """
        Masks the shadow in the image using the specified potential algorithms.

        Args:
            potential (str or list, optional): Potential shadow detection algorithm(s) to use.
                If None, both "unet" and "flood" algorithms will be used. Defaults to None.
            false_cloud_layer (int, optional): The false layer value. Defaults to 0.
        Returns:
            None

        Raises:
            None
        """
        if (potential is None) or (potential.lower() == "both"):
            potential = ["UNet", "Flood"] # port to include the UNet shadow detection in the future
        else:
            potential = [potential]
        # potential shadow mask
        pshadow = np.zeros(self.image.obsmask.shape, dtype="uint8")
        for ialg in potential:
            if ialg.lower() == "flood":
                if C.MSG_FULL:
                    print(">>> masking potential cloud shadow by flood-fill")
                shadow_mask_binary = self.mask_shadow_flood(topo=topo, threshold = threshold)
            elif ialg.lower() == "unet":
                pass # TBD
            # add the shadow mask to the potential shadow mask
            pshadow = pshadow + shadow_mask_binary
            if self.show_figure:
                shadow_mask = shadow_mask_binary.copy().astype("uint8")
                shadow_mask[self.image.filled] = (
                    2  # use all list indicating the classes
                )
                utils.show_shadow_mask(
                    shadow_mask, ["nonshadow", "shadow", "filled"], ialg
                )
            del shadow_mask_binary

        # normalize the potential shadow mask into [0, 1], as weitghted sum to compute the similarity between cloud and shadow, not used when only one potential shadow mask is used
        # pshadow = pshadow / len(potential)
        self.shadow = self.physical.match_cloud2shadow(
            self.cloud_object,
            self.cloud_region,
            pshadow,
            self.physical.water,
            thermal_adjust = thermal_adjust
        )
        
        # delete self.cloud_region, self.cloud_object, pshadow
        del self.cloud_region, self.cloud_object, pshadow

    def display_fmask(self, endname = "", path=None, skip=True):
        """display the fmask, with clear, cloud, shadow, and fill"""
        if skip and os.path.isfile(path):
            if C.MSG_FULL:
                print(f">>> {path} exists, skip to generate the figure of Fmask")
            return
        emask = self.ensemble_mask
        # try to load the Fmask layer .tif
        if emask is None:
            emask, _ = utils.read_raster(path.replace(".png", ".tif"))
        utils.show_fmask(emask, endname, path)
        if C.MSG_FULL:
            print(f">>> saved cloud/shadow visualization as PNG file to {path}")
        
    def print_summary(self):
        """
        Summarizes the cloud mask image with color-coded classes, including land, water, snow, shadow, cloud, and filled.

        Returns:
            None
        """
        mask = self.ensemble_mask

        # Count occurrences of each label
        num_obs = np.count_nonzero(mask != C.LABEL_FILL)
        if num_obs == 0:
            print("Summary: No valid observations.")
            return

        num_cloud = np.count_nonzero(mask == C.LABEL_CLOUD)
        num_shadow = np.count_nonzero(mask == C.LABEL_SHADOW)
        num_snow = np.count_nonzero(mask == C.LABEL_SNOW)
        num_clear = num_obs - num_cloud - num_shadow - num_snow # saving the time to count the clear pixels
        # num_clear = np.count_nonzero((mask == C.LABEL_LAND) | (mask == C.LABEL_WATER))
        # Print summary in one line with formatted percentages
        print("Summary: Cloud = {:.2%}, Shadow = {:.2%}, Snow = {:.2%}, Clear = {:.2%}".format(
            num_cloud / num_obs, num_shadow / num_obs, num_snow / num_obs, num_clear / num_obs))
    
    # %% major port of masking clouds
    def mask_cloud(self, algorithm=None):
        """Masks clouds in the image using the specified algorithm.

        Parameters:
            algorithm (str): The algorithm to use for cloud masking.
                Valid options are "physical", "randomforest", "unet", and "interaction".
                Defaults to "physical".

        Returns:
            it will update the cloud mask in the object
        """
        # if the algorithm is not provided, use the default algorithm
        if algorithm is None:
            algorithm = self.algorithm

        # mask cloud by the specified algorithm
        if algorithm == "physical":
            self.mask_cloud_physical()
        elif algorithm =="lightgbm":
            self.mask_cloud_lightgbm()
        elif algorithm == "unet":
            self.mask_cloud_unet()
        elif algorithm == "interaction":
            self.mask_cloud_interaction()

    def __init__(self, image_path: str = "", algorithm: str = "interaction", base: str = "unet", tune: str = "lightgbm", loadmodel: bool = True, dcloud = 0, dshadow = 5, dsnow = 0):
        """
        Initialize the Fmask object.

        Parameters:
        - image_path (str): The path to the image file.
        - algorithm (str): The algorithm to be used for cloud masking. lightgbm, unet
        - base (str): The base machine learning model to be used. Default is "unet".
        - tune (str): The machine learning model to be used for tuning. lightgbm, unet

        Returns:
        - None
        """
        # set the package directory, which is the parent directory of the current file, as the root, to access the base pre-trained models
        self.dir_package = Path(__file__).parent.parent

       # Initialize image object containing base information on this image
        image_name = Path(image_path).stem
        if image_name.startswith("L"):
            self.image = Landsat(image_path)
        elif image_name.startswith("S"):
            self.image = Sentinel2(image_path)
        
        # the dilated size of cloud, shadow, and snow in pixels (its resolution is the same as the image's resolution)
        self.buffer_cloud = dcloud
        self.buffer_shadow = dshadow # the buffer size of shadow in pixels, which is larger than the original size, 3 by 3 pixels, since the larger dilation size is able better to fill the holes caused by the projection of clouds (to match shadow)
        self.buffer_snow = dsnow

        # which algorithm will be used for cloud masking
        self.algorithm = algorithm
        self.set_base_machine_learning(base)
        self.set_tune_machine_learning(tune)
        
        # init modules that will be used in the cloud masking
        if self.image is not None: # init modules only when the image is loaded
            self.init_modules(loadmodel = loadmodel)