#!/bin/bash

rm npys
ls *npy > npys
while read -r line; do
	python3 convert_numpy.py $line
done < npys
