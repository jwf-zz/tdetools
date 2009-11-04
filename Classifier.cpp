/*
 * Classifier.cpp
 *
 *  Created on: 2009-06-30
 *      Author: jfrank8
 */

#include <stdlib.h>
#include <vector>
#include <omp.h>
#include <ANN/ANN.h>

#include "ClassifyTrajectory.h"
#include "Classifier.h"

#define MATCH_STEPS 32
#define NEIGHBOURS 4

using namespace std;

Classifier::Classifier(vector<NamedModel*> models) {
	this->models = models;
}

Classifier::~Classifier() {
	// TODO Auto-generated destructor stub
}

void Classifier::go(double** data, ulong length) {
	// For each set of MATCH_STEPS points, compute the likelihood under each model.
	TDEModel* model;
	uint M = models.size();
	char* model_name;
	double *projected_data;
	ANNpointArray ap;
	ANNidx nn_idx[NEIGHBOURS+1];
	ANNdist dists[NEIGHBOURS+1];
	CvMat *navg[M], *navg_next[M], *proj_next[M];
	CvMat p, *mdists, *nn[M], *nnn[M];
	ulong i,j,k,l,a,N,pcaembdim;
	double dist, dist_next, *dst, *p1, *p2, *p3, *p4;
	mdists = cvCreateMat(length-MATCH_STEPS-1,models.size(),CV_64FC1);
	cvZero(mdists);

	for (i = 0; i < M; i++) {
		pcaembdim = models[i]->model->getPCAEmbDim();
		navg[i] = cvCreateMat(1,pcaembdim,CV_64FC1);
		navg_next[i] = cvCreateMat(1,pcaembdim,CV_64FC1);
		proj_next[i] = cvCreateMat(1,pcaembdim, CV_64FC1);
		nn[i] = cvCreateMat(NEIGHBOURS,pcaembdim, CV_64FC1);
		nnn[i] = cvCreateMat(NEIGHBOURS,pcaembdim, CV_64FC1);
	}

	for (i = 0; i < length-MATCH_STEPS-1; i++) { //=MATCH_STEPS) {
		// Get the MATCH_STEPS points
		// cout << "Step " << i << endl;
		for (k = 0; k < M; k++) {
			model = models[k]->model;
			model_name = models[k]->name;
			N = model->getLength();
			pcaembdim = model->getPCAEmbDim();
			//projected_data = model->projectData(data+i*embdim,MATCH_STEPS+1,embdim);
			get_ann_points(ap, data[k]+i*pcaembdim, MATCH_STEPS+1, pcaembdim);

			for (j = 0; j < MATCH_STEPS; j++) {
				p = cvMat(1,pcaembdim,CV_64FC1, ap[j]);// &projected_data[j*pcaembdim]);
				// cout << "Checking point: ";
				// print_matrix(&p);

				model->getKNN(ap[j], NEIGHBOURS+1, nn_idx, dists);
				for (l = 0; l < NEIGHBOURS; l++) {
					// Make sure none of the first NEIGHBOURS neighbours is N
					if (nn_idx[l] == ANN_NULL_IDX) break;
					else if ((ulong)nn_idx[l] == N-1) nn_idx[l] = nn_idx[NEIGHBOURS];
					p1 = (double*)(nn[k]->data.ptr+l*nn[k]->step);
					p2 = (double*)(nnn[k]->data.ptr+l*nnn[k]->step);
					p3 = model->getDataPoint(nn_idx[l]);
					p4 = model->getDataPoint(nn_idx[l]+1);
					for (a = 0; a < pcaembdim; a++) {
						*p1++ = *p3++;
						*p2++ = *p4++;
					}
				}
				if (l < NEIGHBOURS) cout << "Warning: Couldn't find enough neighbours." << endl;

				cvReduce(nn[k], navg[k], 0, CV_REDUCE_AVG );
				dist = -sqrt(annDist(pcaembdim,(double*)p.data.ptr,(double*)navg[k]->data.ptr));

				cvReduce(nnn[k], navg_next[k], 0, CV_REDUCE_AVG );
				p1 = (double*)navg_next[k]->data.ptr;
				p2 = (double*)navg[k]->data.ptr;
				dst = (double*)proj_next[k]->data.ptr;
				for (l = 0; l < pcaembdim; l++) {
					*dst++ = ap[j][l] + (*p1++ - *p2++);
				}
				p = cvMat(1,pcaembdim,CV_64FC1, ap[j+1]);
				dist_next = -sqrt(annDist(pcaembdim,(double*)p.data.ptr,(double*)proj_next[k]->data.ptr));
				/*
				cout << "Expected next point: "; print_matrix(proj_next);
				cout << "Real next point: "; print_matrix(&p);
				cout << model_name << ". Dist1: " << dist << ", Dist2: " << dist_next <<  endl;
				*/
				cvmSet(mdists,i,k,cvmGet(mdists,i,k)+(dist+dist_next));
			}
			//cout << model_name << ": " << cvmGet(mdists,i,k) << endl;
		}
	}
	print_matrix(mdists);
	for (i = 0; i < M; i++) {
		cvReleaseMat(&navg[i]);
		cvReleaseMat(&navg_next[i]);
		cvReleaseMat(&nn[i]);
		cvReleaseMat(&nnn[i]);
	}
	cvReleaseMat(&mdists);
}
