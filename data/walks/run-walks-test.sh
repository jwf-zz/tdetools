#!/bin/bash

M=$1
P=$2
D=$3

../../buildtree -c 2 -m $M -p $P -d $D jordan.dat
../../buildtree -c 2 -m $M -p $P -d $D cosmin.dat
../../buildtree -c 2 -m $M -p $P -d $D rob.dat
../../buildtree -c 2 -m $M -p $P -d $D amin.dat
../../classifytrajectory -c 2 classifying-raw.csv > /tmp/walks-out.dat
