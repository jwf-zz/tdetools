#!/bin/bash

M=$1
P=$2
D=$3

./buildtree -m $M -p $P -d $D data/sophia-train.dat && echo "data/sophia-train.dat.dmp > models.ini"
./buildtree -m $M -p $P -d $D data/amin-train.dat && echo "data/amin-train.dat.dmp >> models.ini"
./buildtree -m $M -p $P -d $D data/cosmin-train.dat && echo "data/cosmin-train.dat.dmp >> models.ini"
./buildtree -m $M -p $P -d $D data/robert-train.dat && echo "data/robert-train.dat.dmp >> models.ini"
./classifytrajectory data/walks-test.dat > tmp/walks-out.dat
