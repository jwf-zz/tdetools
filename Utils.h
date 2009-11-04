/*
 * Utils.h
 *
 *  Created on: 2009-06-30
 *      Author: jfrank8
 */

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
	ulong length;
	ulong exclude;
	uint verbosity;
	int delay;
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
} Settings;

void get_embedding(Settings* settings, double*& data, ulong &length);
void get_ann_points(ANNpointArray &dataPts, double* series, ulong rows, ulong cols);
void print_matrix(CvMat* matrix);

#endif /* UTILS_H_ */
