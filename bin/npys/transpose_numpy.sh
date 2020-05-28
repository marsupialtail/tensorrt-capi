#!/bin/bash

rm npys
ls depthwise*weight.npy > npys
while read -r line; do
	python3 transpose_numpy.py $line
done < npys
