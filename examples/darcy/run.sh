#!/bin/bash

res_LF=(32)
theta_LF=(32 64 96)

for i in "${!res_LF[@]}"; do
    for j in "${!theta_LF[@]}"; do
        res="${res_LF[$i]}"
        theta="${theta_LF[$j]}"
        echo "Running config res_LF: $res theta_LF: $theta"
        python gen_data.py --ntheta_LF "$theta" -res_LF "$res" -res_HF 128 --threads 16 --ntheta_HF 128 -n 6000 &> "theta_${theta}_res_${res}.txt"
    done
done
