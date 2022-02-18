#!/bin/bash

for i in {1..8}
do
	for j in {1..56}
	do
		for x in {1..56}
		do
			python3 modify_csv.py $1 $i $j $x
		done
	done
done

