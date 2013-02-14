Time-Delay Embedding Classification Apps
========================================

License: MIT

This project contains proof-of-concept tools for time-series classification using time-delay embeddings (See Frank et al., 2010). 

System Dependencies
===================

1. `sudo apt-get install cmake libopencv-dev libann`

This project depends on the nonlinear time series analysis toolbox [Tisean
3.0.1](http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/)

For convenience, we extracted the necessary C library under the `external/`
folder and should be built automatically.

Compiling
=========

Under the `tdetools/` directory:

1. `cmake .`
2.`make`

Or to to keep the build under a separate directory:

1. `mkdir build`
2. `cmake ../`
3. `make`

Usage
=====

To build a model from a time series represented in CSV format, use the `buildtree` command:

    Usage: ./buildtree [options] [datafile]
    Options:
    Everything not being a valid option will be interpreted as a possible datafile.
    If no datafile is given stdin is read. Just - also means stdin
        -l # of data [default: whole file]
        -x # of rows to ignore [default: 0]
        -M num. of columns to read [default: 1]
        -c columns to read [default: 1]
        -m dimension [default: 2]
        -p # dimension to reduce to using PCA [default: same as m, ie. no reduction]
        -d delay [default: 1]
        -V verbosity level [default: 1]
                0='only panic messages'
                1='+ input/output messages'
        -o output file [default: datafile.dmp]
        -h show these options


To classify a data set, put the filenames of the models, one per line, in a file called `models.ini` and run the `classifytrajectory` command:
    Usage: ./classifytrajectory [options]
    Options:
    Everything not being a valid option will be interpreted as a possible datafile.
    If no datafile is given stdin is read. Just - also means stdin
        -l # of data [default: whole file]
        -x # of rows to ignore [default: 0]
        -M num. of columns to read [default: 1]
        -c columns to read [default: 1]
        -n number of neighbours to match [default: 4]
        -s length of matched segments [default: 32]
        -A # of algorithm to use (1: old, 2: new (unpublished), 3: fancy new(buggy)) [default: old algorithm (1)]
        -V verbosity level [default: 0]
                0='only panic messages'
                1='+ input/output messages'
        -h show these options

The `classifytrajectory` command will send the similarity scores to `STDOUT`, where the order of the scores corresponds to the order of the models in the `models.ini` file.

Included is a script, `run-walks-test.sh`, which gives a simple example of how to use these two applications. It takes three parameters, the embedding dimension, the noise-reduced dimension, and the delay, then uses these parameters to build four models for data from four different people walking. Finally, it classifies a test data file, storing the scores in tmp/walks-out.dat. Running this with values M=5, P=5, D=2, for example, and plotting the values for each column of the output file, you should see that the each of the subjects is recognized, in turn.

If you use this code in your research, we ask you to cite:
> Jordan Frank, Shie Mannor, and Doina Precup. Activity and Gait Recognition with Time-Delay Embeddings. AAAI. 2010.

    @inproceedings{frank10actrec
    author = {J. Frank and S. Mannor and D. Precup},
    title = {Activity and Gait Recognition with Time-Delay Embeddings},
    booktitle = {Proceedings of the Twenty-Fourth AAAI Conference on Artificial Intelligence (AAAI 2010)},
    location = {Atlanta, Georgia, USA},
    year = {2010},
    }

This is proof-of-concept code, but was implemented with efficiency in mind, and has been used to classify large data sets. Please report any bugs to <jordan.frank@cs.mcgill.ca>. Feel free to email me if you'd like statically compiled x86 Linux binaries.
