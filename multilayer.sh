#!/bin/sh

for i in 2 3 4 5 6 7 8
do
   python multilayer.py --dataset LastFM --data-path ./data --nhids 26 --proj-dim 12  --epochs 100  --tuned --n-layers $i
done
