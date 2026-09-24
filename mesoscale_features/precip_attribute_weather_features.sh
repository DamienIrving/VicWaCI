#!/bin/bash
#PBS -P xv83
#PBS -q normalbw
#PBS -l walltime=1:00:00
#PBS -l mem=40GB
#PBS -l ncpus=6
#PBS -l storage=gdata/xv83+gdata/rt52+gdata/xp65
#PBS -v y1,m1,y2,m2

# qsub -v y1=1980,m1=01,y2=2022,m2=12 precip_attribute_weather_features.sh

module use /g/data/xp65/public/modules
module load conda/analysis3-25.11
cd /g/data/xv83/as3189/vicwaci/
python /g/data/xv83/as3189/vicwaci/precip_attribute_weather_features.py -y1 ${y1} -y2 ${y2} -m1 ${m1} -m2 ${m2}