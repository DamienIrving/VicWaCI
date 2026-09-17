#!/bin/bash
#PBS -P xv83
#PBS -q normal
#PBS -l walltime=30:00:00
#PBS -l mem=40GB
#PBS -l ncpus=6
#PBS -l storage=gdata/xv83+gdata/rt52+gdata/xp65
#PBS -v MIN,y1,y2

# qsub -v MIN=30,y1=1980,y2=2022 detect_cyclones.sh
module use /g/data/xp65/public/modules
module load conda/analysis3-25.11
python /g/data/xv83/as3189/vicwaci/detect_cyclones.py --min ${MIN} -y1 ${y1} -y2 ${y2}