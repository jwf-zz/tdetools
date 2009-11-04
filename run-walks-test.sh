#!/bin/bash

M=$1
P=$2
D=$3

./buildtree -m $M -p $P -d $D data/sophia-train.dat
./buildtree -m $M -p $P -d $D data/amin-train.dat
./buildtree -m $M -p $P -d $D data/cosmin-train.dat
./buildtree -m $M -p $P -d $D data/robert-train.dat
./classifytrajectory data/walks-test.dat > /tmp/walks-out.dat
