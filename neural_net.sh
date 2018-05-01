#!/bin/bash
#try different combos of parameters

#epsilon loop
for e in .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0
do
	#alpha loop
	for a in .1 .2 .3 .4 .5 .6 .7 .8 .9 1.0
	do
		#gamma loop
		for g in .7 .8 .9 1.
		do
			#size loop
			for s in 100 200 300 400
			do
				#iterations loop
				for i in 400 600 800 1000
				do
					python neural_net.py $e $a $g $s $i
				done
			done
		done
	done
done	
