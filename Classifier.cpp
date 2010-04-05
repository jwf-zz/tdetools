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

using namespace std;

Classifier::Classifier(vector<NamedModel*> models) {
	this->models = models;
}

Classifier::~Classifier() {
	// TODO Auto-generated destructor stub
}

void Classifier::go(ANNcoord* data, ulong length, ulong embdim,
		uint neighbours, uint seglength) {
	// For each set of MATCH_STEPS points, compute the likelihood under each model.
	TDEModel* model;
	uint M = models.size();
	char* model_name;
	ANNcoord *projected_data;
	ANNpointArray ap;
	ANNidx nn_idx[neighbours+1];
	ANNdist dists[neighbours+1], mdist;
	CvMat *navg[M], *navg_next[M], *proj_next[M];
	CvMat p, np, *mdists, *nn[M], *nnn[M];
	ulong i,j,k,l,a,N,pcaembdim;
	ANNdist dist, *dst, l1, l2; //, dist_next;
	ANNcoord *p1, *p2, *p3, *p4;
	mdists = cvCreateMat(length-seglength-1,models.size(),MAT_TYPE);
	cvZero(mdists);

	cerr << "Using " << neighbours << " neighbours with segment length " << seglength << endl;


	for (i = 0; i < M; i++) {
		pcaembdim = models[i]->model->getPCAEmbDim();
		navg[i] = cvCreateMat(1,pcaembdim,MAT_TYPE);
		navg_next[i] = cvCreateMat(1,pcaembdim,MAT_TYPE);
		proj_next[i] = cvCreateMat(1,pcaembdim, MAT_TYPE);
		nn[i] = cvCreateMat(neighbours,pcaembdim, MAT_TYPE);
		nnn[i] = cvCreateMat(neighbours,pcaembdim, MAT_TYPE);
	}

	for (i = 0; i < length-seglength-1; i++) { //=MATCH_STEPS) {
		// Get the MATCH_STEPS points
		// cout << "Step " << i << endl;
		for (k = 0; k < M; k++) {
			model = models[k]->model;
			model_name = models[k]->name;
			N = model->getLength();
			pcaembdim = model->getPCAEmbDim();
			projected_data = model->projectData(data+i*embdim,seglength+1,embdim);
			get_ann_points(ap, projected_data, seglength+1, pcaembdim);
			mdist = 0.0;
			for (j = 0; j < seglength; j++) {
				p = cvMat(1,pcaembdim,MAT_TYPE, ap[j]);
				model->getKNN(ap[j], neighbours+1, nn_idx, dists);

				for (l = 0; l < neighbours; l++) {
					// Make sure none of the first neighbours is N
					if (nn_idx[l] == ANN_NULL_IDX) break;
					else if ((ulong)nn_idx[l] == N-1) nn_idx[l] = nn_idx[neighbours];
					p1 = (ANNcoord*)(nn[k]->data.ptr+l*nn[k]->step);
					p2 = (ANNcoord*)(nnn[k]->data.ptr+l*nnn[k]->step);
					p3 = model->getDataPoint(nn_idx[l]);
					p4 = model->getDataPoint(nn_idx[l]+1);
					for (a = 0; a < pcaembdim; a++) {
						*p1++ = *p3++;
						*p2++ = *p4++;
					}
				}
				if (l < neighbours) cout << "Warning: Couldn't find enough neighbours." << endl;

				// Computes the mean of the nearest neighbours.
				cvReduce(nn[k], navg[k], 0, CV_REDUCE_AVG );

				// Computes the mean of the neigbours' successors
				cvReduce(nnn[k], navg_next[k], 0, CV_REDUCE_AVG );

				/*
                dist = -sqrt(annDist(pcaembdim, (ANNcoord*) p.data.ptr,
                                (ANNcoord*) navg[k]->data.ptr));
				 */

				p1 = (ANNcoord*)navg_next[k]->data.ptr;
				p2 = (ANNcoord*)navg[k]->data.ptr;
				dst = (ANNcoord*)proj_next[k]->data.ptr;
				for (l = 0; l < pcaembdim; l++) {
					*dst++ = ap[j][l] + (*p1++ - *p2++);
				}

				// np is the subsequent point in the trajectory to be classified.
				np = cvMat(1, pcaembdim, MAT_TYPE, ap[j + 1]);

				/*
                                  dist_next = -sqrt(annDist(pcaembdim,
                                  (ANNcoord*) np.data.ptr,
                                  (ANNcoord*) proj_next[k]->data.ptr));
				 */
				//mdist = mdist + (dist + dist_next);

				// Shift each vector to the origin and compute the dot product
				// normalized by the length of the larger vector.
				p1 = (ANNcoord*)p.data.ptr;
				p2 = (ANNcoord*)np.data.ptr;
				p3 = (ANNcoord*)proj_next[k]->data.ptr;
				dist = 0.0;
				l1 = 0.0; // Length of first vector
				l2 = 0.0; // Length of second vector
				for (l = 0; l < pcaembdim; l++) {
					dist = dist + (*p2 - *p1)*(*p3 - *p1);
					l1 = l1 + (*p2 - *p1)*(*p2 - *p1);
					l2 = l2 + (*p3 - *p1)*(*p3 - *p1);
					*p1++; *p2++; *p3++;
				}
				//l1 = sqrt(l1);
				//l2 = sqrt(l2);
				mdist = mdist + dist/MAX(l1,l2);
			}
			annDeallocPts(ap);
			// cvmSet(mdists, i, k, mdist);
			if (k > 0) cout << " ";
			cout << mdist;
		}
		cout << "\n";
	}
	//print_matrix(mdists);
	for (i = 0; i < M; i++) {
		cvReleaseMat(&navg[i]);
		cvReleaseMat(&navg_next[i]);
		cvReleaseMat(&nn[i]);
		cvReleaseMat(&nnn[i]);
	}
	cvReleaseMat(&mdists);
}
