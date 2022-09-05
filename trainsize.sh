#!/bin/bash
for i in 1 3 5 7 10 13 15 17 20 23 25 27 30
do
    python performance_trainsize.py --dataset LastFM --data-path ./data --model DGCN --nhids 26 --proj-dim 12  --epochs 100 --tuned --num-train-nodes $i
done