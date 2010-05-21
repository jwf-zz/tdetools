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
//#include <opencv/cvaux.h>

#include "Utils.h"
#include "BuildTree.h"
#include "TDEModel.h"

using namespace std;
using namespace boost;

TDEModel::TDEModel(Settings* settings) {
    ANNcoord *data, *projecteddata;

    length = settings->length;
    embdim = settings->embdim;
    delay = settings->delay;
    use_pca = settings->pcaembset;

    get_embedding(settings, data, length);

    if (use_pca) {
    	cout << "Computing PCA bases.\n";
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
	cerr << "Model params: " << delay << " " << " " << embdim << " " << avgsize << endl;
	if (avgsize > 0) {
		use_pca = 1;
		avg = cvCreateMat(1,avgsize,MAT_TYPE);
		ANNcoord* ptr = (ANNcoord*)avg->data.ptr;
		for (i = 0; i < avgsize; i++) {
			*model_file >> *ptr++;
		}
	}
	else {
		use_pca = 0;
		avg = NULL;
		bases = NULL;
	}
	*model_file >> basesrows >> basescols;
	if (use_pca) {
		bases = cvCreateMat(basesrows,basescols,MAT_TYPE);
		ANNcoord* ptr = (ANNcoord*)bases->data.ptr;
		for (i = 0; i < basesrows; i++) {
			for (j = 0; j < basescols; j++) {
				*model_file >> *ptr++;
			}
		}
	}

	kdTree = new ANNkd_tree(*model_file);
	dataPts = kdTree->thePoints();
	length = kdTree->nPoints();

	cerr << "Loaded " << length << " points." << endl;
	model_file->close();
	delete model_file;
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
    		fout << " " << CV_MAT_ELEM(*avg, ANNcoord, 0, i);
    	}
    }
	fout << endl;
    if (bases == NULL) {
    	fout << "0 0" << endl;
    }
    else {
    	fout << bases->rows << " " << bases->cols << endl;
    	for (int i = 0; i < bases->rows; i++) {
        	fout << CV_MAT_ELEM(*bases, ANNcoord, i, 0);
    		for (int j = 1; j < bases->cols; j++) {
    			fout << " " << CV_MAT_ELEM(*bases, ANNcoord, i, j);
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
    uint neighbours = 4;
	// +1 in case one of the neighbours is the last point in the model.
    nn_idx = new ANNidx[neighbours+1];
    dists = new ANNdist[neighbours+1];

    for (i = 0; i < dim; i++) {
    	trajectory[0][i] = s0[i];
    }

    for (i = 1; i < N; i++) {
    	getKNN(trajectory[i-1], neighbours+1, nn_idx, dists);
    	for (j = 0; j < dim; j++) {
    		trajectory[i][j] = 0.0;
    		for (k = 0; k < neighbours; k++) {
    			if (nn_idx[k] == ANN_NULL_IDX) break;
    			else if (nn_idx[k] == (int)length-1) nn_idx[k] = nn_idx[neighbours];
    			trajectory[i][j] += dataPts[nn_idx[k]+1][j];
    		}
    		trajectory[i][j] = trajectory[i][j] / (ANNcoord)k + generator();
    	}
    }
}

ANNpoint TDEModel::getDataPoint(uint idx) {
	return dataPts[idx];
}

void TDEModel::computePCABases(ANNcoord *data, uint rows, uint cols, uint numbases) {
	CvMat **embedding, *cov, *eigenvectors, *eigenvalues, *vector;
	ANNcoord *basesdata;
	uint i, j, offset;

	cov = cvCreateMat(cols, cols, MAT_TYPE);
	eigenvectors = cvCreateMat(cols,cols,MAT_TYPE);
	eigenvalues = cvCreateMat(cols,cols,MAT_TYPE);
	embedding = new CvMat*[rows];
	for (i = 0; i < rows; i++) {
		vector = cvCreateMatHeader(1, cols, MAT_TYPE);
		cvInitMatHeader(vector, 1, cols, MAT_TYPE, data + i * cols);
		embedding[i] = vector;
	}
	avg = cvCreateMat(1,cols,MAT_TYPE);
	cvCalcCovarMatrix((const CvArr **)embedding, rows, cov, avg, CV_COVAR_NORMAL);
	cvSVD(cov, eigenvalues, eigenvectors, 0, CV_SVD_MODIFY_A);

	basesdata = new ANNcoord[cols*numbases];
	for (i = 0, offset = 0; i < cols; i++) {
		for (j = 0; j < numbases; j++) {
			basesdata[offset++] = ((ANNcoord*)eigenvectors->data.ptr)[i*cols+j];
		}
	}
	bases = cvCreateMatHeader(cols, numbases, MAT_TYPE);
	cvInitMatHeader(bases, cols, numbases, MAT_TYPE, basesdata);

	for (i = 0; i < rows; i++) {
    	cvReleaseMat(&embedding[i]);
    }
    delete [] embedding;
    cvReleaseMat(&cov);
    cvReleaseMat(&eigenvectors);
    cvReleaseMat(&eigenvalues);
}

ANNcoord* TDEModel::projectData(ANNcoord* data, uint rows, uint cols) {
	if (!use_pca) return data;
	ANNcoord* shifteddata = new ANNcoord[rows*cols];
	ANNcoord* projecteddata = new ANNcoord[rows*bases->cols];
	uint i, j, offset;

	for (i = 0, offset = 0; i < rows; i++) {
		for (j = 0; j < cols; j++) {
			shifteddata[offset] = data[offset] - ((ANNcoord*)avg->data.ptr)[j];
			offset++;
		}
	}
	CvMat projected = cvMat(rows, bases->cols, MAT_TYPE, projecteddata);
	CvMat dataMat = cvMat(rows, cols, MAT_TYPE, shifteddata);
	cvGEMM(&dataMat, bases, 1.0, NULL, 0.0, &projected, 0);
	/*
	for (i = 0; i < (unsigned)bases->cols; i++) {
		cout << projecteddata[i] << " " ;
	}
	cout << endl;
	*/
	delete [] shifteddata;
	return projecteddata;
}
