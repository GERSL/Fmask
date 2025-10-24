"""
Author: Shi Qiu, Zhe Zhu
Email: shi.qiu@uconn.edu, zhe@uconn.edu
Affiliation: University of Connecticut
Date: 2025-10-23
Version: 5.0.1
License: MIT

Website: https://github.com/gersl/fmask

Description:
Batch processing of Landsat and Sentinel-2 images using Fmask 5. See details at fmask.py

Usage examples:
- python fmask.py --model=UPL --dcloud=3 --dshadow=5 --imagedir='/gpfs/sharedfs1/zhulab/Shi/ProjectCloudDetectionFmask5/HLSDataset'

Command-line arguments:
--model: the Fmask cloud detection model to use (PHY, GBM, UNT, LPL, UPU, LPU, UPL)
--dcloud: dilation for cloud mask in pixels; default is 3
--dshadow: dilation for shadow mask in pixels; default is 5
--dsnow: dilation for snow mask in pixels; default is 0
--output: destination directory for results; if not provided, results are saved in the image's directory
--skip_existing: skip processing the image when its fmask layer exists; default is 'no'
--save_metadata: save the model metadata to a CSV file; default is 'no'
--display_fmask: display the Fmask result and save it as a PNG file; default is 'no'
--display_image: display the color composite images (NIR-Red-Green and SWIR1-NIR-Red) and save them as PNG files; default is 'no'
--print_summary: print the summary of the Fmask result, including the percentage of cloud, shadow, snow, and clear; default is 'no'

Changelog:
- 5.0.1 (2025-10-23): 
    Algorithms described in detail by Qiu et al., 2026.
    Compared to 5.0.0, mainly added the Sen2Cloud+ dataset for ML training, and reduced image chip size from 512×512 to 256×256 to accommodate this dataset, and shifted image chips to maximize valid pixel coverage and mitigate edge effects in UNet models.

- 5.0.0 (2025-06-05): Initial release.
"""


import os
import sys
from pathlib import Path
import click
import glob
from pathlib import Path
sys.path.append(
    str(Path(__file__).parent.parent.joinpath("src"))
)
from fmask import fmask_physical, fmask_lightgbm, fmask_unet, fmask_lpl,  fmask_lpu, fmask_upl, fmask_upu

@click.command()
@click.option("--model", "-m", type=str, help="Cloud detection model to use.", default="UPL")
@click.option("--dcloud", "-c", type=int, help="Dilation for cloud mask in pixels", default=0)
@click.option("--dshadow", "-s", type=int, help="Dilation for shadow mask in pixels", default=5)
@click.option("--dsnow", "-n", type=int, help="Dilation for shadow mask in pixels", default=0)
@click.option(
    "--imagedir", "-i",
    type=str,
    help="Directory containing Landsat/Sentinel-2 images. Supports multiple images.",
    default="/scratch/shq19004/CMIX2/Phase2/multi-temporal/S2/L1C/equatorial_guinea",
)
@click.option(
    "--output", "-o",
    type=str,
    help="Destination directory for results. If not provided, results are saved in the resource directory.",
    default="/scratch/shq19004/CMIX2/Phase2Fmask/S2/L1C/equatorial_guinea",
)
@click.option("--skip_existing", "-s", type=click.Choice(["yes", "no", "Yes", "No", "YES", "NO"]), help="Skip processing if results already exist (set to 0 to force generation).", default="yes")
@click.option("--save_metadata", "-md", type=click.Choice(["yes", "no", "Yes", "No", "YES", "NO"]), help="Save model metadata to a CSV file.", default="yes")
@click.option("--display_fmask", "-df", type=click.Choice(["yes", "no", "Yes", "No", "YES", "NO"]), help="Display and save the Fmask result as a PNG file.", default="yes")
@click.option("--display_image", "-di", type=click.Choice(["yes", "no", "Yes", "No", "YES", "NO"]), help="Display and save color composite images as PNG files.", default="yes")
@click.option("--print_summary", "-ps", type=click.Choice(["yes", "no", "Yes", "No", "YES", "NO"]), help="Print Fmask summary including cloud, shadow, snow, and clear percentages.", default="no")
@click.option("--ci", "-ci", type=int, help="The core's id", default=1)
@click.option("--cn", "-cn", type=int, help="The number of cores", default=1)
def main(model, dcloud, dshadow, dsnow, imagedir, output, skip_existing, save_metadata, display_fmask, display_image, print_summary, ci, cn) -> None:

    skip_existing = True if skip_existing.lower() == "yes" else False
    save_metadata = True if save_metadata.lower() == "yes" else False
    display_fmask = True if display_fmask.lower() == "yes" else False
    display_image = True if display_image.lower() == "yes" else False
    print_summary = True if print_summary.lower() == "yes" else False
    
    # Create  image list
    path_image_list  = sorted(glob.glob(os.path.join(imagedir, '[L|S]*')))
    # Only keep the image folders
    path_image_list = [path_image for path_image in path_image_list if os.path.isdir(path_image)]
    
    # Divide the tasks into different cores
    path_image_list = [path_image_list[i] for i in range(ci - 1, len(path_image_list), cn)] # ci - 1 is the index
    print(f"Core {ci}/{cn}: Processing a total of {len(path_image_list)} images")
    
    # Loop through the images
    for path_image in path_image_list:
        print()  # Prints a new line before the progress update
        image_name = Path(path_image).stem
        if "OPER_PRD_" in image_name:
            print(f"{image_name} is not a valid image")
            print(">>> skipping...")
            continue
        if ("LO08_" in image_name) or ("LO09_" in image_name):
            print(f"{image_name} is not a valid image (lack of thermal band)")
            print(">>> skipping...")
            continue
        # temporially support the second image folder
        # if '.SAFE' in path_image:
        #    path_image = os.path.join(path_image, image_name + '.SAFE')
        
        # setup the output directory as none and use the default image path if not provided
        if output == "":
            output = None

        # check the model name and run the corresponding model
        if model.upper() == 'UPL': # UNet-Physical-LightGBM recommnad for landsat 8-9 and sentinel 2
            fmask_upl(path_image = path_image, dcloud=dcloud, dshadow=dshadow, dsnow=dsnow,
                    destination = output, skip = skip_existing, 
                    metadata = save_metadata, display_fmask = display_fmask, display_image = display_image, print_summary = print_summary)
        elif model.upper() == 'LPL': # LightGBM-Physical-LightGBM recommnad for landsat 4-7
            fmask_lpl(path_image = path_image, dcloud=dcloud, dshadow=dshadow, dsnow=dsnow,
                    destination = output, skip = skip_existing, 
                    metadata = save_metadata, display_fmask = display_fmask, display_image = display_image, print_summary = print_summary)
        elif model.upper() == 'PHY': # Fmask 4.6
            fmask_physical(path_image = path_image, dcloud=dcloud, dshadow=dshadow, dsnow=dsnow,
                    destination = output, skip = skip_existing, 
                    metadata = save_metadata, display_fmask = display_fmask, display_image = display_image, print_summary = print_summary)
        elif model.upper() == 'GBM': # LightGBM
            fmask_lightgbm(path_image = path_image, dcloud=dcloud, dshadow=dshadow, dsnow=dsnow,
                    destination = output, skip = skip_existing, 
                    metadata = save_metadata, display_fmask = display_fmask, display_image = display_image, print_summary = print_summary)
        elif model.upper() == 'UNT': # UNet
            fmask_unet(path_image = path_image, dcloud=dcloud, dshadow=dshadow, dsnow=dsnow,
                    destination = output, skip = skip_existing, 
                    metadata = save_metadata, display_fmask = display_fmask, display_image = display_image, print_summary = print_summary)
        elif model.upper() == 'UPU': # UNet-Physical-UNet
            fmask_upu(path_image = path_image, dcloud=dcloud, dshadow=dshadow, dsnow=dsnow,
                    destination = output, skip = skip_existing, 
                    metadata = save_metadata, display_fmask = display_fmask, display_image = display_image, print_summary = print_summary)
        elif model.upper() == 'LPU': # LightGBM-Physical-UNet
            fmask_lpu(path_image = path_image, dcloud=dcloud, dshadow=dshadow, dsnow=dsnow,
                    destination = output, skip = skip_existing, 
                    metadata = save_metadata, display_fmask = display_fmask, display_image = display_image, print_summary = print_summary)
        else:
            print(f"Model {model} is not supported.")
            return

# main port to run the fmask by command line
if __name__ == "__main__":
    main()