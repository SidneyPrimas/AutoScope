#!/bin/bash
####################################
#
# Train particles_ML_model.py
#
####################################

echo "Initiate training of particles_ML_model.py"

# Create Logging and Readme File
dest="./log"
NOWT=$(date +"D%F-T%T")
LOGFILE="log-$NOWT.log"
READFILE="readme-$NOWT.txt"
echo "LOG at $LOGFILE"
echo "REAME at $READFILE"

readme="Important Parameters:"

echo "$readme" > "$dest/$READFILE"

count=1
# Train model indefinitely (until manually interrupted)
while [ $count -eq $count ]
do
	start=$(date +"D%F-T%T")
	echo "Iteration $count: Start at $start"
	echo "Iteration $count: Start at $start" >> "$dest/$LOGFILE"

	# Run Actual Training Cycle
	frameworkpython ./particles/particles_ML_model.py  >> "$dest/$LOGFILE"

	end=$(date +"D%F-T%T")
	echo "Iteration $count: End at $end"
	count=$(( $count + 1 ))
done