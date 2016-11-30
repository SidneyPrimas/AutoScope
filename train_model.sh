#!/bin/bash
####################################
#
# Train particles_ML_model.py
#
####################################

# Note: Currently, in order to run this, we need to runthis file from gitroot. 

# ToDo: 

echo "Initiate training of particles_ML_model.py"

# Create Logging and Readme File
dest="./log"
NOWT=$(date +"D%F-T%T")
LOGFILE="$dest/log-$NOWT.log"
READFILE="$dest/readme-$NOWT.txt"

# 
echo "LOG at $LOGFILE"
echo "REAME at $READFILE"

readme="Important Parameters:"

echo "$readme" > "$READFILE"

count=1
# Train model indefinitely (until manually interrupted)
while [ $count -eq $count ]
do
	start=$(date +"D%F-T%T")
	echo "Iteration $count: Start at $start"
	echo "Iteration $count: Start at $start" >> "$LOGFILE"

	# Run Actual Training Cycle
	frameworkpython ./particles/particles_ML_model.py  >> "$LOGFILE"

	end=$(date +"D%F-T%T")
	echo "Iteration $count: End at $end"
	count=$(( $count + 1 ))
done