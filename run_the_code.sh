#!/bin/bash
distance_type='discret_frechet'
data_type='sroma'
GPU=0
rho=7
threshold=0.01
d=128
timestamp=$(date +%m_%d_%H)
python -u train.py --distance_type ${distance_type} --data_name ${data_type} --GPU ${GPU} --d ${d} --rho ${rho} --threshold ${threshold} 2>&1 | tee ./logs/${distance_type}/${timestamp}_${data_type}_${distance_type}_${rho}_${threshold}_${d}.txt