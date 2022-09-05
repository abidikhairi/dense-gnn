#!/bin/sh

for i in {1..10}
do
	python linkprediction.py --dataset Computers --data-path ./data --nhids 26 --proj-dim 12\
	--epochs 300 --skip-connection concat --tuned
done
