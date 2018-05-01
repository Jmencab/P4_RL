#!/bin/bash
#try different combos of parameters

#epsilon loop
for e in .5 .6 .7 .8 .9 1.0
do
	#alpha loop
	for a in .5 .6 .7 .8 .9 1.0
	do
		#gamma loop
		for g in  .9 1.
		do
			#size loop
			for s in 100 200 300 
			do
				python neural_net.py $e $a $g $s 500

			done
		done
	done
done	
