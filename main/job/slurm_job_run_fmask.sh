#!/bin/bash
#SBATCH -J fmask
#SBATCH --partition=priority-gpu
#SBATCH --account=sas18043
#SBATCH --qos=sas18043a100
#SBATCH --constraint=a100
#SBATCH --time=240:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --array 1-10
#SBATCH -o log/%x-out-%A_%4a.out
#SBATCH -e log/%x-err-%A_%4a.err


. "/home/shq19004/miniconda3/etc/profile.d/conda.sh"  # startup conda
conda activate fmask 

echo $SLURMD_NODENAME # display the node name
cd ../


DCLOUD=3
DSHADOW=5
SAVEMETA='yes'
DISPLAYFMASK='yes'
DISPLAYIMG='yes'

FOLDER_SRC='/scratch/shq19004/ProjectCloudDetectionFmask5/Validation/Landsat89' # 1350 cores in total
FOLDER_DES='/scratch/shq19004/ProjectCloudDetectionFmask5/Validation/Landsat89FmaskDilate0GPU' # end names will be provided afterward
MODEL='UPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK

FOLDER_SRC='/scratch/shq19004/ProjectCloudDetectionFmask5/Validation/Sentinel2' # 1350 cores in total
FOLDER_DES='/scratch/shq19004/ProjectCloudDetectionFmask5/Validation/Sentinel2FmaskDilate0GPU' # end names will be provided afterward
MODEL='UPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK

FOLDER_SRC='/scratch/shq19004/ProjectCloudDetectionFmask5/Validation/Landsat47' # 1350 cores in total
FOLDER_DES='/scratch/shq19004/ProjectCloudDetectionFmask5/Validation/Landsat47FmaskDilate0GPU' # end names will be provided afterward
MODEL='LPL'
python fmask_batch.py --model=$MODEL --imagedir=$FOLDER_SRC --output=$FOLDER_DES --dcloud=$DCLOUD --dshadow=$DSHADOW --save_metadata=$SAVEMETA --ci=$SLURM_ARRAY_TASK_ID --cn=$SLURM_ARRAY_TASK_MAX --display_image=$DISPLAYIMG --display_fmask=$DISPLAYFMASK


echo 'Finished!'
exit
