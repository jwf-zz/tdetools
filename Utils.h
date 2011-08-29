//============================================================================
// Name        : Utils.h
// Author      : Jordan Frank (jordan.frank@cs.mcgill.ca)
// Copyright   : MIT
// Description : Some helpful utility methods.
//============================================================================

#include <stdlib.h>
#include <ANN/ANN.h>
#include <opencv/cxcore.h>

#ifndef ulong
#define ulong unsigned long
#endif
#ifndef uint
#define uint unsigned int
#endif

#ifndef UTILS_H_
#define UTILS_H_

typedef struct {
	unsigned long length;
	unsigned long exclude;
	uint verbosity;
	int delay;
	uint neighbours;
	uint seglength;
	uint indim;
	uint embdim;
	uint pcaembdim;
	char *column;
	char *infile;
	char *outfile;
	char dimset;
	char embset;
	char pcaembset;
	char delayset;
	char stdo;
	uint algorithm;
} Settings;

void get_embedding(Settings* settings, ANNcoord*& data, unsigned long &length);
void get_ann_points(ANNpointArray &dataPts, ANNcoord* series, unsigned long rows, uint cols);
void print_matrix(CvMat* matrix);

#define MAT_TYPE CV_64FC1
#define FLOAT_SCAN "%G"
#define FLOAT_OUT "%.8G"

#endif /* UTILS_H_ */
