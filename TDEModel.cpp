/*
 * TDEModel.cpp
 *
 *  Created on: 2009-06-29
 *      Author: jfrank8
 */

#include <stdlib.h>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <Tisean/tsa.h>
#include <ANN/ANN.h>
#include <opencv/cxcore.h>
#include <boost/random.hpp>
#include <opencv/cvaux.h>

#include "Utils.h"
#include "BuildTree.h"
#include "TDEModel.h"

#define NEIGHBOURS 4

using namespace std;
using namespace boost;

TDEModel::TDEModel(Settings* settings) {
    double *data, *projecteddata;

    length = settings->length;
    embdim = settings->embdim;
    delay = settings->delay;
    use_pca = settings->pcaembset;

    get_embedding(settings, data, length);

    if (use_pca) {
    	computePCABases(data, length, embdim, settings->pcaembdim);
    	projecteddata = projectData(data, length, embdim);
    	delete [] data;
    	data = projecteddata;
    }
    else {
    	avg = NULL;
    	bases = NULL;
    }
    /*
	for (uint i = 0; i < (unsigned)bases->cols; i++) {
		cout << data[i] << " " ;
	}
	cout << endl;
	*/
    get_ann_points(dataPts, data, length, settings->pcaembdim);
    kdTree = new ANNkd_tree(dataPts,length,settings->pcaembdim);
    settings->length = length;
    delete [] data;
}

TDEModel::TDEModel(ifstream* model_file) {
	int avgsize, basesrows, basescols, i, j;
	*model_file >> delay;
	*model_file >> embdim;
	*model_file >> avgsize;
	if (avgsize > 0) {
		use_pca = 1;
		avg = cvCreateMat(1,avgsize,CV_64FC1);
		double* ptr = (double*)avg->data.ptr;
		for (i = 0; i < avgsize; i++) {
			*model_file >> *ptr++;
		}
	}
	else {
		avg = NULL;
	}
	*model_file >> basesrows >> basescols;
	if (basesrows > 0 && basescols > 0) {
		use_pca = 1;
		bases = cvCreateMat(basesrows,basescols,CV_64FC1);
		double* ptr = (double*)bases->data.ptr;
		for (i = 0; i < basesrows; i++) {
			for (j = 0; j < basescols; j++) {
				*model_file >> *ptr++;
			}
		}
	}

	kdTree = new ANNkd_tree(*model_file);
	dataPts = kdTree->thePoints();
	length = kdTree->nPoints();
}

TDEModel::~TDEModel() {
	if (avg != NULL) cvReleaseMat(&avg);
	if (bases != NULL) cvReleaseMat(&bases);
	delete kdTree;
	delete [] dataPts;
}

void TDEModel::DumpTree(char* outfile) {
    ofstream fout(outfile);
    fout << delay << endl;
    fout << embdim << endl;
    if (avg == NULL) {
    	fout << 0 << endl;
    }
    else {
    	fout << avg->cols << endl;
    	for (int i = 0; i < avg->cols; i++) {
    		fout << " " << CV_MAT_ELEM(*avg, double, 0, i);
    	}
    }
	fout << endl;
    if (bases == NULL) {
    	fout << "0 0";
    }
    else {
    	fout << bases->rows << " " << bases->cols << endl;
    	for (int i = 0; i < bases->rows; i++) {
        	fout << CV_MAT_ELEM(*bases, double, i, 0);
    		for (int j = 1; j < bases->cols; j++) {
    			fout << " " << CV_MAT_ELEM(*bases, double, i, j);
    		}
    		fout << endl;
    	}
    }
    kdTree->Dump(ANNtrue, fout);
    fout.close();
}

void TDEModel::getKNN(ANNpoint ap, uint k, ANNidxArray nn_idx, ANNdistArray dists) {
	kdTree->annkSearch(ap, k, nn_idx, dists);
//    for (uint i = 0; i < k; i++) {
//            cout << "Point " << i+1 << ": [" << dataPts[nn_idx[i]][0] << "," << dataPts[nn_idx[i]][1] << "," << dataPts[nn_idx[i]][2] << "], Dist: " << sqrt(dists[i]) << endl;
//    }
}

void TDEModel::simulateTrajectory(ANNpoint s0, ANNpointArray trajectory, uint dim, ulong N) {
    ANNidxArray nn_idx;
    ANNdistArray dists;
    uint i,j,k;
    variate_generator<mt19937, normal_distribution<> > generator(mt19937(time(0)), normal_distribution<>(0.0,0.1));

	// +1 in case one of the neighbours is the last point in the model.
    nn_idx = new ANNidx[NEIGHBOURS+1];
    dists = new ANNdist[NEIGHBOURS+1];

    for (i = 0; i < dim; i++) {
    	trajectory[0][i] = s0[i];
    }

    for (i = 1; i < N; i++) {
    	getKNN(trajectory[i-1], NEIGHBOURS+1, nn_idx, dists);
    	for (j = 0; j < dim; j++) {
    		trajectory[i][j] = 0.0;
    		for (k = 0; k < NEIGHBOURS; k++) {
    			if (nn_idx[k] == ANN_NULL_IDX) break;
    			else if (nn_idx[k] == (int)length-1) nn_idx[k] = nn_idx[NEIGHBOURS];
    			trajectory[i][j] += dataPts[nn_idx[k]+1][j];
    		}
    		trajectory[i][j] = trajectory[i][j] / (double)k + generator();
    	}
    }
}

ANNpoint TDEModel::getDataPoint(uint idx) {
	return dataPts[idx];
}

void TDEModel::computePCABases(double *data, uint rows, uint cols, uint numbases) {
	CvMat **embedding, *cov, *eigenvectors, *eigenvalues, *vector;
	double *basesdata;
	uint i, j, offset;

	cov = cvCreateMat(cols, cols, CV_64FC1);
	eigenvectors = cvCreateMat(cols,cols,CV_64FC1);
	eigenvalues = cvCreateMat(cols,cols,CV_64FC1);
	embedding = new CvMat*[rows];
	for (i = 0; i < rows; i++) {
		vector = cvCreateMatHeader(1, cols, CV_64FC1);
		cvInitMatHeader(vector, 1, cols, CV_64FC1, data + i * cols);
		embedding[i] = vector;
	}
	avg = cvCreateMat(1,cols,CV_64FC1);
	cvCalcCovarMatrix((const CvArr **)embedding, rows, cov, avg, CV_COVAR_NORMAL);
	cvSVD(cov, eigenvalues, eigenvectors, 0, CV_SVD_MODIFY_A);

	basesdata = new double[cols*numbases];
	for (i = 0, offset = 0; i < cols; i++) {
		for (j = 0; j < numbases; j++) {
			basesdata[offset++] = ((double*)eigenvectors->data.ptr)[i*cols+j];
		}
	}
	bases = cvCreateMatHeader(cols, numbases, CV_64FC1);
	cvInitMatHeader(bases, cols, numbases, CV_64FC1, basesdata);

	for (i = 0; i < rows; i++) {
    	cvReleaseMat(&embedding[i]);
    }
    delete [] embedding;
    cvReleaseMat(&cov);
    cvReleaseMat(&eigenvectors);
    cvReleaseMat(&eigenvalues);
}

double* TDEModel::projectData(double* data, uint rows, uint cols) {
	if (!use_pca) return data;
	double* shifteddata = new double[rows*cols];
	double* projecteddata = new double[rows*bases->cols];
	uint i, j, offset;

	for (i = 0, offset = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			shifteddata[offset] = data[offset] - ((double*)avg->data.ptr)[j];
			offset++;
		}
	}
	CvMat projected = cvMat(rows, bases->cols, CV_64FC1, projecteddata);
	CvMat dataMat = cvMat(rows, cols, CV_64FC1, shifteddata);
	cvGEMM(&dataMat, bases, 1.0, NULL, 0.0, &projected, 0);
	/*
	for (i = 0; i < (unsigned)bases->cols; i++) {
		cout << projecteddata[i] << " " ;
	}
	cout << endl;
	*/
	return projecteddata;
}
