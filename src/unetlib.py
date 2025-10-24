"""UNet to mask cloud or cloud shadow"""

# pylint: disable=not-callable
# pylint: disable=line-too-long
# pylint: disable=no-member
import os
import numpy as np
import random
from osgeo import gdal_array
import torch
from torch import from_numpy
from torch.utils.data import DataLoader, Dataset as BaseDataset
import segmentation_models_pytorch as smp
from pathlib import Path
from utils import normalize_datacube, init_patch_offanchors
import time
import constant as C
np.seterr(invalid='ignore') # ignore the invalid errors


class Dataset(BaseDataset):
    """Dataset class for the UNet model

    Args:
        BaseDataset (Class): Base dataset class
    """

    @property
    def directory_images(self):
        """
        Returns the path to the directory containing the images.

        Returns:
            str: The path to the directory containing the images.
        """
        return os.path.join(self.directory, "images")

    @property
    def directory_labels(self):
        """
        Returns the path to the directory containing the labels.

        Returns:
            str: The path to the directory containing the labels.
        """
        return os.path.join(self.directory, "labels")
    @property
    def length(self):
        """
        Returns the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.
        """
        return len(self.ids)

    def __getitem__(self, index):
        if self.dformat == "npy":
            image = np.load(self.images_fps[index])
            label = np.load(self.labels_fps[index])
        elif self.dformat == "tif":
            image = gdal_array.LoadFile(self.images_fps[index])  # label starts from 0
            label = gdal_array.LoadFile(self.labels_fps[index])  # label starts from 0

        # print(self.images_fps[index])
        # print(image.shape)
        # print(label.shape)

        if self.layerindices is None:
            image = from_numpy(image)
        else:
            image = from_numpy(image[self.layerindices, :, :])  # transform as tensor

        # change the label based on the classes with index as the value, starting from 0
        _label = label.copy()
        for icls, cls in enumerate(self.classes):
            if cls == "noncloud":
                label[_label == C.LABEL_CLEAR] = icls
                label[_label == C.LABEL_SHADOW] = icls
                if "filled" not in self.classes: # in case if we consider filled as noncloud
                    label[_label == C.LABEL_FILL] = icls
            elif cls == "cloud":
                label[_label == C.LABEL_CLOUD] = icls
            elif cls == "nonshadow":
                label[_label == C.LABEL_CLEAR] = icls
                label[_label == C.LABEL_CLOUD] = icls
            elif cls == "shadow":
                label[_label == C.LABEL_SHADOW] = icls
            elif cls == "filled":
                label[_label == C.LABEL_FILL] = icls
        label = from_numpy(label)  # transform as tensor

        return image, label

    def set_data_index(self, value):
        """Set the layer indices

        Returns:
            None
        """
        self.layerindices = value

    def __init__(
        self,
        directory,
        patch_size,
        classes,
        layerindices=None,
        number=float("inf"),
        random_seed=None,
        exclude=None,
    ):
        super(Dataset, self).__init__()

        self.directory = directory
        self.patch_size = patch_size
        self.classes = classes
        self.layerindices = layerindices  # index of the layer that will be used for CNN

        # find the files available in the folder
        self.ids = os.listdir(self.directory_labels)  # all files will be loaded

        _num_images = len(self.ids)
        # exclude the files that are not needed if exclude is provided
        if exclude is not None:
            if isinstance(exclude, str):
                exclude = [exclude]
            self.ids = [
                img
                for img in self.ids
                if not any(ex in img for ex in exclude)
            ]
            # print the exclude information
            if C.MSG_FULL:
                for ex in exclude:
                    print(f">>> excluding {ex} from the training dataset")
                print(f">>> having {len(self.ids)} patches ({_num_images - len(self.ids)} excluded)")
        else:
            if C.MSG_FULL:
                print(f">>> having on {len(self.ids)} patches")
        del _num_images

        # check the number of images
        # the format of the dataset
        self.dformat = os.path.splitext(self.ids[0])[1][1:]  # npy or tif
        # min num
        self.number = min(number, len(self.ids))
        # random shuffle
        random.seed(random_seed)
        self.ids = random.sample(self.ids, self.number)

        # make the paths of all images and the labels
        _images_dir = self.directory_images
        _labels_dir = self.directory_labels
        self.images_fps = [os.path.join(_images_dir, chip_id) for chip_id in self.ids]
        self.labels_fps = [os.path.join(_labels_dir, chip_id) for chip_id in self.ids]

    def __len__(self):
        return len(self.ids)


class UNet(object):
    """_summary_"""

    # %%Attributes
    image = None  # image object
    path = None  # path of base unet model
    model = None  # unet model
    optimizer = None # optimizer
    predictors = None  # the predictors used in the model, that will be used to extract the datacube from the image loaded
    classes = [
        "noncloud",
        "cloud",
        "filled",
    ]  # represent the values in the label, 0, 1, 2
    backbone = "resnet34"
    decoder_channels = [256, 128, 64, 32]  # decorder_channels
    patch_size = 256
    patch_stride_train = 224  # for generating patches for training
    patch_stride_classify = 224  # for classifying the whole image
    batch_size = 32 # change to 32 for training from the original 16 for 512 pixels
    learn_rate = 1e-3
    epoch = 40  # 40 for Landsat 8 and Sentinel-2
    shuffle = True
    workers = 2
    prefetch = 2
    device = "cuda"  # 'cpu' or 'cuda'
    continue_train = True  # indicate whether to continue training

    # will be used when it is tuned
    tune_epoch = 5
    tune_learn_rate = 1e-4

    # dataset for training
    path_train_dataset = None  # remaining for the patch to trainning unet
    train_dataset: Dataset = None

    def set_train_data_path(self, path_train_dataset: str) -> None:
        """Set the path of the training dataset

        Args:
            path_train_dataset (str): path of the training dataset

        Returns:
            None
        """
        self.path_train_dataset = path_train_dataset

    def set_patch_size(self, patch_size: int) -> None:
        """Set the patch size

        Args:
            patch_size (int): patch size

        Returns:
            None
        """
        self.patch_size = patch_size

    def set_patch_stride_train(self, patch_stride_train: int) -> None:
        """Set the patch stride for trainning

        Args:
            patch_stride_train (int): patch stride for trainning

        Returns:
            None
        """
        self.patch_stride_train = patch_stride_train

    def set_patch_stride_classify(self, patch_stride_classify: int) -> None:
        """Set the patch stride for classification

        Args:
            patch_stride_classify (int): patch stride for classification

        Returns:
            None
        """
        self.patch_stride_classify = patch_stride_classify

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch

        Args:
            epoch (int): epoch

        Returns:
            None
        """
        self.epoch = epoch

    @property
    def activated(self):
        """Check the model is activated or not

        Returns:
            bool: True for activated, False for not activated
        """
        return self.model is not None

    @property
    def basename(self):
        """
        Get the base name of the unet by looping through the classes and taking the first 3 characters of each class.

        Returns:
            str: The base name of the unet.
        """
        _basename = "".join([cls[0] for cls in self.classes])
        _basename = f"unet_{_basename}"  # unet_ncf
        return _basename

    # %%Methods
    def init_model(self) -> None:
        """Initialize the model"""
        self.model = smp.Unet(
            encoder_name=self.backbone,
            encoder_depth=len(self.decoder_channels),
            encoder_weights=None,
            decoder_channels=self.decoder_channels,
            decoder_use_batchnorm=True,  # BatchNorm2d layer between Conv2D and Activation layers is used.
            decoder_attention_type=None,
            in_channels=len(self.predictors),  # according to the predictors
            classes=len(self.classes),
            activation=None,
        ).to(
            self.device
        )  # make it on GPU or CPU

    def init_train_dataset(self, path: str = None, exclude=None, patch_index = None) -> None:
        """
        Initializes the training dataset.

        Args:
            path (str, optional): Path to the training dataset. If not provided, the default path will be used.
            exclude (list, optional): List of images to exclude from the training dataset.

        Returns:
            None
        """
        if path is None:
            path = self.path_train_dataset
        # Load all training and testing dataset
        self.train_dataset = Dataset(
            path,  # path to the data
            self.patch_size,  # patch size
            classes=self.classes,  # classes
            exclude=exclude,  # exclude the image that will be used to train the model
            random_seed=C.RANDOM_SEED,
        )
        if patch_index is not None:
            self.train_dataset.set_data_index(patch_index)

    def train(self, path: str = None, save_epochs=None, exclude_loss_filled=True) -> None:
        """
        Trains the UNet model using the specified training dataset.

        Args:
            path (str, optional): The path to save the trained models. If not provided, the default path will be used.
            save_epochs (int or list of int, optional): The epochs at which to save the trained models. If an integer is provided, only the model at that epoch will be saved. If a list of integers is provided, the models at those epochs will be saved. Defaults to None.

        Returns:
            None
        """

        if path is None:
            path = self.path
        if save_epochs is not None:
            # covnert to list
            if isinstance(save_epochs, int):
                save_epochs = [save_epochs]
        else:  # if not provided, save the last model
            save_epochs = [self.epoch]

        Path(path).mkdir(parents=True, exist_ok=True)
        # base name of the unet by looping the classes with first 3 characters
        basename = "".join([cls[0] for cls in self.classes])
        basename = f"unet_{basename}"  # unet_ncf

        # load training data
        if self.train_dataset is None:
            self.init_train_dataset(self.path_train_dataset, exclude=None)
        # obtain the dataset as training with shuffle and test without shuffle
        train_dataset_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            generator=torch.Generator().manual_seed(
                C.RANDOM_SEED
            ),  # to preserve reproducibility
            shuffle=self.shuffle,
            num_workers=self.workers,
            pin_memory=True,
            prefetch_factor=self.prefetch,
            persistent_workers=True,
        )

        # unet model initialized if it has not been activated
        if not self.activated:
            self.init_model()

        # Loss function
        if exclude_loss_filled:
            if "filled" in self.classes:
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.classes.index("filled"))
            else:
                loss_fn = torch.nn.CrossEntropyLoss() # if we do not exclude the filled, then we use the default loss function
        else:
            loss_fn = torch.nn.CrossEntropyLoss()
        # Default in 32-bit floating point precision, but this can help that lower precisions, 16-bit,
        # can be significantly faster on modern GPUs
        scaler = torch.cuda.amp.GradScaler()  # training in mixed precision.
        # Optimization can be easily integrated in the future.
        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=float(self.learn_rate)
        )
        # record of the epoch starting
        epoch_start = 1

        # continue to train based on the previous existing record
        epoch_records = []
        if self.continue_train:
            # check out the epoches that have not been saved
            for epoch in range(self.epoch, 0, - 1): # backward
                path_model = os.path.join(path, f"{basename}_{epoch:03d}.pt")
                if os.path.isfile(path_model):
                    if os.stat(path_model).st_size > 10000000: # assume model > 10 mb, 1,000,000 byte to 1 mb
                        epoch_start = epoch + 1 # next epoch will be conducted
                        state_dict = torch.load(path_model)
                        self.model.load_state_dict(state_dict["model_state_dict"])
                        optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                        del state_dict
                        break
            # build the epoch records
            for epoch in range(1, epoch_start):
                path_model = os.path.join(path, f"{basename}_{epoch:03d}.pt")
                epoch_record = torch.load(path_model)
                # remove the models in the records
                epoch_record["model_state_dict"] = None
                epoch_record["optimizer_state_dict"] = None
                epoch_records.append(epoch_record)
                del epoch_record
        # if all models exist, then return
        if epoch_start > self.epoch:
            if C.MSG_FULL:
                print(f">>> model has been trained to the maximum epoch {self.epoch}")
            return
        # Train the model
        if C.MSG_FULL:
            print(f">>> training model-{basename} by {self.device}")
        self.model.train() # Make sure gradient tracking is on, and do a pass over the data
        label_filled = self.classes.index("filled") if "filled" in self.classes else None
        for epoch in range(epoch_start, self.epoch + 1):  # range starts at 1
            # recording the running time in secs
            start_epoch = time.time()
            # msg
            if C.MSG_FULL:
                print(f">>>> epoch {epoch:03d}/{self.epoch:03d}", end="")
            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train()
            # model_loss_train, model_overall_train = 0.0, 0.0
            model_loss_train = 0.0
            for data in train_dataset_loader:
                # if C.MSG_FULL:
                #     print(".", end="")
                # data loaded
                image_train, label_train = data
                # Zero your gradients for every batch!
                optimizer.zero_grad()
                # loss
                # loss_train, overall_train = self.evaluate_model(
                #     self.model, loss_fn, image_train, label_train, self.device, label_filled=label_filled
                # ) # filled is excluded
                loss_train, _ = self.evaluate_model(
                    self.model, loss_fn, image_train, label_train, self.device, label_filled=label_filled, overall=False
                ) # filled is excluded
                
                # Scale Gradients
                scaler.scale(loss_train).backward()
                # Adjust learning weights
                scaler.step(optimizer)
                scaler.update()
                # Gather data and report
                model_loss_train += loss_train.item()
                # model_overall_train += overall_train.item()
            train_time = time.time() - start_epoch  # unit: second
            # loss per epoch
            model_loss_train = model_loss_train / len(train_dataset_loader) # average value of the loss against the number of batches
            # model_overall_train = model_overall_train / len(train_dataset_loader) # average value of the loss against the number of batches
            model_loss_test = np.nan  # preserve the nan value Place holder
            model_overall_test = np.nan  # preserve the nan value Place holder
            model_overall_train = np.nan  # preserve the nan value Place holder
            # msg
            if C.MSG_FULL:
                # print(
                #     f" with loss = {model_loss_train:0.4f} & overall = {model_overall_train:.4f} in {train_time/60:03.2f} mins"
                # )
                print(
                    f" with loss = {model_loss_train:0.4f} in {train_time/60:03.2f} mins"
                )

            # record the information of each epoch
             # record the information of each epoch
            epoch_record = {
                "predictors": self.predictors,
                "classes": self.classes,
                "backbone": self.backbone,
                "decoder_channels": self.decoder_channels,
                "patch_size": self.patch_size,
                "patch_stride_train": self.patch_stride_train,
                "batch_size": self.batch_size,
                "learn_rate": self.learn_rate,
                "epoch": epoch,
                "shuffle": self.shuffle,
                "workers": self.workers,
                "prefetch": self.prefetch,
                "device": self.device,
                "continue_train": self.continue_train,
                "loss_train": model_loss_train,
                "loss_test": model_loss_test,
                "path_train_data": self.path_train_dataset,
                "number_train_data": self.train_dataset.length,
                "overall_train": model_overall_train,
                "overall_test": model_overall_test,
                "train_time": train_time,
            }
            epoch_records.append(epoch_record)

            # saving train processing
            # save entire model first, with .part
            path_model = os.path.join(path, f"{basename}_{epoch:03d}.pt")
            # append model at the end of the epoch_record
            epoch_record["model_state_dict"] = self.model.state_dict()
            epoch_record["optimizer_state_dict"] = optimizer.state_dict()
            torch.save(epoch_record, path_model + ".part")
            os.rename(path_model + ".part", path_model) # avoid abrupt during saving

            # delete the previous epoch's model because it is too large
            if (epoch > 1) and (epoch - 1 not in save_epochs):  # delete the data of the previous one (but store the basic training process info) if which is not asked to be saved
                path_model_pre = os.path.join(path, f"{basename}_{epoch-1:03d}.pt")
                epoch_record_pre = epoch_records[epoch - 2].copy() # - 2, as index, is to the previous one
                epoch_record_pre["model_state_dict"] = None
                epoch_record_pre["optimizer_state_dict"] = None
                torch.save(epoch_record_pre, path_model_pre + ".part")
                os.rename(path_model_pre + ".part", path_model_pre)
                del epoch_record_pre

    def load_model(self) -> None:
        """Load model from the path

        Returns:
            None
        """
        #if self.model is None:
        state_dict = torch.load(self.path, map_location=torch.device(self.device))
        # load the key values
        self.predictors = state_dict["predictors"] # setup the input layers, recorded by the state_dict
        self.classes = state_dict["classes"]
        self.backbone = state_dict["backbone"]
        self.decoder_channels = state_dict["decoder_channels"]
        self.patch_size = state_dict["patch_size"]
        self.patch_stride_train = state_dict["patch_stride_train"]
        
        self.init_model()
        self.model.load_state_dict(state_dict["model_state_dict"])
        # optimizer for tuning
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=float(self.tune_learn_rate)
        )
        self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])

    def classify(self, probability="none", normalize=True, shift=True) -> tuple:
        """Classify the image by the model and return the class and probability

        Args:
            probability (srt, optional): One item of the classes, or "none" (means not to extract the prob layer), or "default" (highest score for the classified results). Defaults to "none".
            normalize (bool, optional): Whether to normalize the datacube. Defaults to True.
            shift (bool, optional): Whether to shift the patch off anchors. Defaults to True.
        Returns:
            image_class, image_prob: class result (0: noncloud, 1: cloud) and probability result
        """
        # check the probability index
        prob_index = (
            -1 if probability == "default" else
            -2 if probability == "none" else
            self.classes.index(probability)
        )

        # rescale the date cube
        datacube = self.image.data.get(self.predictors)
        if normalize:
            datacube = normalize_datacube(datacube, obsmask=self.image.obsmask)
    
        if C.MSG_FULL:
            print(">>> classifying image by unet")

        # Convert the datacube to tensor if it is not
        if not isinstance(datacube, torch.Tensor):
            datacube = torch.from_numpy(datacube).to(
                self.device, dtype=torch.float
            )  # (chanel, height, width)
    
        offanchors = init_patch_offanchors(self.image.obsmask, size=self.patch_size, stride=self.patch_stride_classify, shift=shift)
        # Change the r c off
        wincut = int((self.patch_size - self.patch_stride_classify) / 2)
   
        # initlized image_class and image_prob

        h, w = datacube.shape[1], datacube.shape[2]
        image_class = np.full((h, w), 255, dtype=np.uint8) # 255 means no data, will be filled later using image_class_full
        image_class_full = np.zeros((h, w), dtype=np.uint8)
        if prob_index == -2:
            image_prob = None  # no need to initialize
        else:
            image_prob = np.zeros((h, w), dtype=np.float32)
            image_prob_full = np.zeros((h, w), dtype=np.float32)
        self.model.eval()
        
        for ibatch in range(0, len(offanchors), self.batch_size):
            # obtain the batch anchors
            batch_anchors = offanchors[ibatch : np.min((ibatch + self.batch_size, len(offanchors)))]  # maximum batch size
            # obtain the batch tensor of the datacube
            # batch_tensor = torch.stack([datacube[:, r:r+self.patch_size, c:c+self.patch_size] for r, c, _, _ in batch_anchors])
            # make the classification
            with torch.no_grad(): # one line to finish the classification, in order to save the memory
                batch_probs = torch.nn.functional.softmax(self.model(torch.stack([datacube[:, r:r+self.patch_size, c:c+self.patch_size] for r, c, _, _ in batch_anchors])), dim=1)
            # batch processing the class and prob layer
            if prob_index == -2: # -2 means the top class will be extracted by this model, but not the probability
                _, batch_classes = batch_probs.topk(1, dim=1)  # top label with top prob.
                batch_classes = batch_classes.to("cpu").numpy()
                for j, (r_off, c_off, _, _) in enumerate(batch_anchors):
                    image_class[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = batch_classes[
                        j,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_class_full[
                        r_off : r_off + self.patch_size, c_off : c_off + self.patch_size
                    ] = batch_classes[
                        j,
                        0,
                        :,
                        :
                    ]
            elif prob_index == -1: # -1 means only the predicted class will be extracted by this model
                batch_probs, batch_classes = batch_probs.topk(1, dim=1)  # top label with top prob.
                batch_probs = batch_probs.to("cpu").numpy()
                batch_classes = batch_classes.to("cpu").numpy()
                for j, (r_off, c_off, _, _) in enumerate(batch_anchors):
                    image_class[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = batch_classes[
                        j,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_class_full[
                        r_off : r_off + self.patch_size, c_off : c_off + self.patch_size
                    ] = batch_classes[
                        j,
                        0,
                        :,
                        :
                    ]
                    image_prob[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = batch_probs[
                        j,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_prob_full[
                        r_off: r_off + self.patch_size, c_off: c_off + self.patch_size
                    ] = batch_probs[
                        j,
                        0,
                        :,
                        :
                    ]
            elif prob_index >= 0: # the class is the classified class, and the prob is the probability of the defined index
                _, batch_classes = batch_probs.topk(1, dim=1)  # # (N, 1, H, W)
                # select the prob of the defined class and index based on the classes, but keep same shape
                batch_probs = batch_probs[:, prob_index, :, :].to("cpu").numpy() # indicate the probability of the defined class
                batch_classes = batch_classes.to("cpu").numpy()  # (N, 1, H, W)
                for j, (r_off, c_off, _, _) in enumerate(batch_anchors):
                    image_class[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = batch_classes[
                        j,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_class_full[
                        r_off : r_off + self.patch_size, c_off : c_off + self.patch_size
                    ] = batch_classes[
                        j,
                        0,
                        :,
                        :
                    ]
                    image_prob[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = batch_probs[
                        j,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_prob_full[
                        r_off: r_off + self.patch_size, c_off: c_off + self.patch_size
                    ] = batch_probs[
                        j,
                        :,
                        :
                    ]
        # fill the 255
        # Fill missing values from full map
        missing_mask = (image_class == 255)
        image_class[missing_mask] = image_class_full[missing_mask]
        if image_prob is not None:
            image_prob[missing_mask] = image_prob_full[missing_mask]
        return image_class, image_prob
    
    def classify_backup(self, probability="none", normalize=True, shift=True) -> tuple:
        """Classify the image by the model and return the class and probability

        Args:
            probability (srt, optional): One item of the classes, or "none" (means not to extract the prob layer), or "default" (highest score for the classified results). Defaults to "none".
            normalize (bool, optional): Whether to normalize the datacube. Defaults to True.
            shift (bool, optional): Whether to shift the patch off anchors. Defaults to True.
        Returns:
            image_class, image_prob: class result (0: noncloud, 1: cloud) and probability result
        """
        # check the probability index
        prob_index = (
            -1 if probability == "default" else
            -2 if probability == "none" else
            self.classes.index(probability)
        )

        # rescale the date cube
        datacube = self.image.data.get(self.predictors)
        if normalize:
            datacube = normalize_datacube(datacube, obsmask=self.image.obsmask)
    
        if C.MSG_FULL:
            print(">>> classifying image by unet")

        # Convert the datacube to tensor if it is not
        if not isinstance(datacube, torch.Tensor):
            datacube = torch.from_numpy(datacube).to(
                self.device, dtype=torch.float
            )  # (batch, chanel, height, width)
        datacube = torch.unsqueeze(
            datacube, dim=0
        )  # add one more dimension for the batch
        offanchors = init_patch_offanchors(self.image.obsmask, size=self.patch_size, stride=self.patch_stride_classify, shift=shift)
        # Change the r c off
        wincut = int((self.patch_size - self.patch_stride_classify) / 2)
   
        # initlized image_class and image_prob

        h, w = datacube.shape[2], datacube.shape[3]
        image_class = np.full((h, w), 255, dtype=np.uint8) # 255 means no data, will be filled later using image_class_full
        image_class_full = np.zeros((h, w), dtype=np.uint8)
        if prob_index == -2:
            image_prob = None  # no need to initialize
        else:
            image_prob = np.zeros((h, w), dtype=np.float32)
            image_prob_full = np.zeros((h, w), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():  # disabled gradient calculation
            for offanchor in offanchors:
                r_off, c_off = offanchor[0], offanchor[1]
                # classify the pacth
                patchprob = torch.nn.functional.softmax(
                    self.model(
                        datacube[
                            :,
                            :,
                            r_off : r_off + self.patch_size,
                            c_off : c_off + self.patch_size,
                        ]
                    ),
                    dim=1,
                )

                # extract the prob and class
                if (
                    prob_index >= 0
                ):  # prob_class_index is the index of the class that we want to extract the probability
                    _, patchclass = patchprob.topk(
                        1, dim=1
                    )  # top label with top prob.
                    # select the prob of the defined class and index based on the classes, but keep same shape
                    patchprob = patchprob[:, prob_index, :, :].to("cpu").numpy()
                    # patchprob = patchprob.to("cpu").numpy()
                    image_class[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = patchclass[
                        0,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_class_full[
                        r_off : r_off + self.patch_size, c_off : c_off + self.patch_size
                    ] = patchclass[
                        0,
                        0,
                        :,
                        :
                    ]
                    image_prob[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = patchprob[
                        0,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_prob_full[
                        r_off: r_off + self.patch_size, c_off: c_off + self.patch_size
                    ] = patchprob[
                        0,
                        0,
                        :,
                        :
                    ]
                elif (
                    prob_index == -1
                ):  # -1 means only the predicted class will be extracted by this model
                    patchprob, patchclass = patchprob.topk(
                        1, dim=1
                    ).to("cpu").numpy()  # top label with top prob.
                    # patchprob = patchprob.to("cpu").numpy()
                    image_prob[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = patchprob[
                        0,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_prob_full[
                        r_off: r_off + self.patch_size, c_off: c_off + self.patch_size
                    ] = patchprob[
                        0,
                        0,
                        :,
                        :
                    ]
                elif (
                    prob_index == -2
                ):  # -2 means the top class will be extracted by this model, but not the probability
                    _, patchclass = patchprob.topk(
                        1, dim=1
                    )  # top label with top prob.

                    # which will be extracted all the time
                    patchclass = patchclass.to("cpu").numpy()
                    image_class[
                        r_off + wincut: r_off + self.patch_size - wincut, c_off + wincut: c_off + self.patch_size - wincut
                    ] = patchclass[
                        0,
                        0,
                        wincut: self.patch_size - wincut,
                        wincut: self.patch_size - wincut,
                    ]
                    image_class_full[
                        r_off : r_off + self.patch_size, c_off : c_off + self.patch_size
                    ] = patchclass[
                        0,
                        0,
                        :,
                        :
                    ]
        # fill the 255
        # Fill missing values from full map
        missing_mask = (image_class == 255)
        image_class[missing_mask] = image_class_full[missing_mask]
        if image_prob is not None:
            image_prob[missing_mask] = image_prob_full[missing_mask]
        return image_class, image_prob

    def check_device(self) -> None:
        """examine device, CPU or GPU

        Returns:
            str: the device
        """

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.has_mps:  # mac's gpu
            self.device = "mps"
        else:  # cpu
            self.device = "cpu"

    def get_patch_off(self, obsmask, fetch_boundary=False):
        """Extract the list of patch off based on the full datacube and mmask

        Args:
            obsmask (2d array): Extent of observation
            fetch_boundary (bool, optional): Fetch the patch at the end. Defaults to False.

        Returns:
            2d array: 2d array of patch offset, i.e., row and column
        """
        # cut the datacube into patches
        # Change the r c off
        rows_off = np.arange(
            0, obsmask.shape[0] - self.patch_size + 1, self.patch_stride_train
        )
        cols_off = np.arange(
            0, obsmask.shape[1] - self.patch_size + 1, self.patch_stride_train
        )

        # fetch the last chips at the border
        if fetch_boundary:
            rows_off = np.unique(
                np.append(rows_off, obsmask.shape[0] - self.patch_size)
            )
            cols_off = np.unique(
                np.append(cols_off, obsmask.shape[1] - self.patch_size)
            )

        patchs_off = np.ones((len(rows_off) * len(cols_off), 2), dtype=np.int16)
        i_off = 0
        for r_off in rows_off:
            for c_off in cols_off:
                # check the mask if contains any valid pixels
                if (
                    True
                    in obsmask[
                        r_off : r_off + self.patch_size, c_off : c_off + self.patch_size
                    ]
                ):  # exclude pure dark chips
                    patchs_off[i_off, :] = [r_off, c_off]
                    i_off = i_off + 1
        patchs_off = patchs_off[0:i_off, :]  # exclude the 100% non-valid chip data
        return patchs_off

    def tune(self, reference, normalize=True, exclude_loss_filled = True):
        "Tune the pretrained model based on exisiting full image"
        if self.tune_epoch > 0: # only when we will tune the model by the full image
            if C.MSG_FULL:
                print(">>> tuning unet")
            # rescale the date cube
            if normalize:
                datacube = normalize_datacube(
                    self.image.data.get(self.predictors), obsmask=self.image.obsmask
                )

            # Convert the datacube and reference to tensor
            datacube = torch.from_numpy(datacube).to(self.device, dtype=torch.float)
            reference = torch.from_numpy(reference).to(self.device, dtype=torch.float)
            # Loss function
            if exclude_loss_filled:
                if "filled" in self.classes:
                    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.classes.index("filled"))
                else:
                    loss_fn = torch.nn.CrossEntropyLoss() # if we do not exclude the filled, then we use the default loss function
            else:
                loss_fn = torch.nn.CrossEntropyLoss()
            # Default in 32-bit floating point precision, but this can help that lower precisions, 16-bit,
            # can be significantly faster on modern GPUs
            scaler = torch.cuda.amp.GradScaler()  # training in mixed precision.

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train()
            # Initialize the loss and overall
            model_loss_train = 0.0
            # Get the patch offset
            patchs_off = self.get_patch_off(self.image.obsmask, fetch_boundary=True)

            # Continue to train based on the previous existing record
            epoch_start = 1
            # Train the model
            for ei in range(epoch_start, self.tune_epoch + 1):  # range starts at 1
                start_epoch = time.time()
                random.seed(ei)  # shuffle at each epoch
                patchs_off = patchs_off[
                    random.sample(
                        np.arange(0, patchs_off.shape[0]).tolist(), patchs_off.shape[0]
                    ),
                    :,
                ]
                for patch_id_start in np.arange(0, patchs_off.shape[0], self.batch_size):
                    # data loaded
                    patch_id_end = min(
                        patch_id_start + self.batch_size, patchs_off.shape[0]
                    )

                    image_train = torch.empty(
                        (
                            patch_id_end - patch_id_start,
                            datacube.shape[0],
                            self.patch_size,
                            self.patch_size,
                        ),
                        dtype=torch.float,
                    )
                    label_train = torch.empty(
                        (patch_id_end - patch_id_start, self.patch_size, self.patch_size),
                        dtype=torch.long,
                    )  # long is for computing the loss

                    for i, pi in enumerate(np.arange(patch_id_start, patch_id_end)):
                        image_train[i, :, :, :] = datacube[
                            :,
                            patchs_off[pi, 0] : patchs_off[pi, 0] + self.patch_size,
                            patchs_off[pi, 1] : patchs_off[pi, 1] + self.patch_size,
                        ]
                        label_train[i, :, :] = reference[
                            patchs_off[pi, 0] : patchs_off[pi, 0] + self.patch_size,
                            patchs_off[pi, 1] : patchs_off[pi, 1] + self.patch_size,
                        ]

                    # Zero your gradients for every batch!
                    self.optimizer.zero_grad()
                    # loss
                    loss_train, _ = self.evaluate_model(
                        self.model,
                        loss_fn,
                        image_train,
                        label_train,
                        self.device,
                        overall=False,
                    )
                    # Scale Gradients
                    scaler.scale(loss_train).backward()
                    # Adjust learning weights
                    scaler.step(self.optimizer)
                    scaler.update()
                    # Gather data and report
                    model_loss_train += loss_train.item()
                train_time = time.time() - start_epoch
                # loss per epoch
                model_loss_train = model_loss_train / len(patchs_off)
                # msg
                if C.MSG_FULL:
                    print(
                        ">>>> epoch {0:02d} with loss of {1:0.3f} in {2:03.0f} secs".format(
                            ei, model_loss_train, train_time
                        )
                    )

        # classify the image
        image_class, image_prob = self.classify(
            probability="default", normalize=normalize
        )
        return image_class, image_prob

    @staticmethod
    def evaluate_model(model, loss_fn, image, label, device, label_filled=None, overall=True):
        """Evaluate the model during trainning process

        Args:
            model (unet): unet model
            loss_fn (function): loss_fn
            image (data bands): image
            label (label): label
            device (string): device

        Returns:
            accuracy_loss, accuracy_over: loss and accuracy
        """
        image = image.to(device, dtype=torch.float)
        # image = torch.nan_to_num(image, nan=0)  # convert nan values as zero for cases
        label = label.to(device, dtype=torch.long)  # long is for computing the loss

        # Enables auto-casting for the forward pass (model + loss)
        if device != "cuda":
            device = "cpu"
        with torch.autocast(device):
            predictions = model(image)  # predicted label
            # total loss for the batch images
            accuracy_loss = loss_fn(
                predictions, label
            )  # loss value of the batch of images

            # overall accuracy for the batch images
            if overall:
                # the label has been changed to the continuous numbers starting from 0, i.e., 0, 1, 2, 3, 4, 5, 6, 7
                label_model = torch.argmax(
                    torch.nn.functional.softmax(predictions, dim=1), dim=1
                )
                if label_filled is not None: # exclude the filled pixels
                    # exclude the loss filled pixels, i.e., 255
                    mask_filled = (label != label_filled)
                    label_model = label_model[mask_filled]
                    label = label[mask_filled]
                # total accuracy for the batch images
                accuracy_over = (label_model == label).sum() / torch.numel(label)
            else:
                accuracy_over = 0
        return accuracy_loss, accuracy_over

    def __init__(
        self,
        classes=None,
        predictors=None,
        learn_rate=1e-3,
        epoch=40,
        patch_size=256,
        patch_stride_train=224,
        patch_stride_classify=224,
        tune_epoch=5,
        tune_learn_rate=1e-4,
        path=None,
    ):
        """Initialize the UNet model

        Args:
            classes (list, optional): Classes sequenced by the values, i.e., 0, 1, 2. Defaults to None.
            predictors (list, optional): Predictors that will be used in the model. Defaults to None.
            learn_rate (number, optional): Learning rate. Defaults to 1e-3.
            epoch (int, optional): Epoch. Defaults to 40.
            patch_size (int, optional): Size of patch in pixels. Defaults to 256.
            patch_stride_train (int, optional): Size of patch stride for trainning in pixels. Defaults to 224.
            patch_stride_classify (int, optional): Size of patch stride for classification in pixels. Defaults to 224.
            path (str, optional): Path of the unet model. Defaults to None.
        """
        # update the attributes that can be varied
        self.classes = classes
        self.predictors = predictors
        self.learn_rate = learn_rate
        self.epoch = epoch
        self.patch_size = patch_size
        self.patch_stride_train = patch_stride_train
        self.patch_stride_classify = patch_stride_classify
        self.tune_epoch = tune_epoch
        self.tune_learn_rate = tune_learn_rate

        # init the model
        self.check_device()  # examine gpu is available or not
        # self.init_model()  # unet model initialized
        if path is not None:
            self.path = path
            self.load_model()  # load the model from the path


# %%
