//============================================================================
// Name        : Classifier.cpp
// Author      : Jordan Frank (jordan.frank@cs.mcgill.ca)
// Copyright   : MIT
// Description : Implements the Geometric Template Matching algorithm for 
//               time-series classification.
//============================================================================

#include <stdlib.h>
#include <vector>
#include <omp.h>
#include <ANN/ANN.h>
#include <iostream>
#include <fstream>

#include "ClassifyTrajectory.h"
#include "Classifier.h"

using namespace std;

#define DEBUGSTEPS 0

Classifier::Classifier(vector<NamedModel*>* models) {
	this->models = models;
}

Classifier::~Classifier() {
	// TODO Auto-generated destructor stub
	this->models = NULL;
}

void Classifier::go(ANNcoord* data, uint length, uint embdim, uint neighbours, uint seglength, uint algorithm, uint verbosity) {
	// For each set of MATCH_STEPS points, compute the likelihood under each model.
	TDEModel* model;
	uint M = models->size();
	uint extra_neighbours = 0;
	uint progress;
	if (algorithm == 2) {
		extra_neighbours = 32;
	}
	else if (algorithm == 3) {
		extra_neighbours = 5;
	}
	char* model_name;
	ANNcoord *projected_data;
	ANNpointArray ap;
	ANNidx nn_idx[neighbours+extra_neighbours+1];
	ANNdist dists[neighbours+extra_neighbours+1], mdist;
	CvMat *navg[M], *navg_next[M], *proj_next[M];
	CvMat p, np, *mdists, *nn[M], *nnn[M];
	uint i,j,k,l,h,a,N,pcaembdim;
	ANNdist dist, *dst, l1, l2; //, dist_next;
	ANNcoord *p1, *p2, *p3, *p4, *p5;
	ANNcoord interpcoeff;
	ofstream debugfile;
	char debugfilename[50];
	mdists = cvCreateMat(length-seglength-1,M,MAT_TYPE);
	cvZero(mdists);

	if (verbosity > 0)
		cerr << "Using " << neighbours << " neighbours with segment length " << seglength << endl;


	for (i = 0; i < M; i++) {
		pcaembdim = (*models)[i]->model->getPCAEmbDim();
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
			model = (*models)[k]->model;
			model_name = (*models)[k]->name;
			N = model->getLength();
			pcaembdim = model->getPCAEmbDim();
			projected_data = model->projectData(data+i*embdim,seglength+1,embdim);
			get_ann_points(ap, projected_data, seglength+1, pcaembdim);
			if (data+i*embdim != projected_data) // If it's really been projected
				delete [] projected_data;
			mdist = 0.0;

			if (algorithm == 1) {
				if (i < DEBUGSTEPS) {
					sprintf(debugfilename, "/tmp/debug/step%d-model%d.dat",i+1,k+1);
					debugfile.open(debugfilename, ios::out | ios::trunc);
				}

				for (j = 0; j < seglength; j++) {
					p = cvMat(1,pcaembdim,MAT_TYPE, ap[j]);
					if (i < DEBUGSTEPS) {
						for (l=0; l < pcaembdim; l++) {
							debugfile << CV_MAT_ELEM(p, ANNcoord, 0, l) << " ";
						}
					}

					// np is the subsequent point in the trajectory to be classified.
					np = cvMat(1, pcaembdim, MAT_TYPE, ap[j + 1]);
					if (i < DEBUGSTEPS) {
						for (l=0; l < pcaembdim; l++) {
							debugfile << CV_MAT_ELEM(np, ANNcoord, 0, l) << " ";
						}
					}

					// Get the indices of the nearest neighbours.
					model->getKNN(ap[j], neighbours+1, nn_idx, dists);

					// Now get the data for these indices
					for (l = 0; l < neighbours; l++) {
						// Make sure none of the first neighbours is N
						if (nn_idx[l] == ANN_NULL_IDX) break;
						else if ((uint)nn_idx[l] == N-1) nn_idx[l] = nn_idx[neighbours];
						p1 = (ANNcoord*)(nn[k]->data.ptr+l*nn[k]->step);
						p2 = (ANNcoord*)(nnn[k]->data.ptr+l*nnn[k]->step);
						p3 = model->getDataPoint(nn_idx[l]);
						p4 = model->getDataPoint(nn_idx[l]+1);
						for (a = 0; a < pcaembdim; a++) {
							*p1++ = *p3++;
							*p2++ = *p4++;
						}
						if (i < DEBUGSTEPS) {
							p1 = (ANNcoord*)(nn[k]->data.ptr+l*nn[k]->step);
							for (h=0; h < pcaembdim; h++) {
								debugfile << p1[h] << " ";
							}
						}
						if (i < DEBUGSTEPS) {
							p2 = (ANNcoord*)(nnn[k]->data.ptr+l*nnn[k]->step);
							for (h=0; h < pcaembdim; h++) {
								debugfile << p2[h] << " ";
							}
						}
					}

					if (l < neighbours) cerr << "Warning: Couldn't find enough neighbours." << endl;

					// Computes the mean of the nearest neighbours.
					cvReduce(nn[k], navg[k], 0, CV_REDUCE_AVG );
					if (i < DEBUGSTEPS) {
						for (l=0; l < pcaembdim; l++) {
							debugfile << CV_MAT_ELEM(*navg[k], ANNcoord, 0, l) << " ";
						}
					}

					// Computes the mean of the neigbours' successors
					cvReduce(nnn[k], navg_next[k], 0, CV_REDUCE_AVG );
					if (i < DEBUGSTEPS) {
						for (l=0; l < pcaembdim; l++) {
							debugfile << CV_MAT_ELEM(*navg_next[k], ANNcoord, 0, l) << " ";
						}
					}

					p1 = (ANNcoord*)navg_next[k]->data.ptr;
					p2 = (ANNcoord*)navg[k]->data.ptr;
					dst = (ANNcoord*)proj_next[k]->data.ptr;
					for (l = 0; l < pcaembdim; l++) {
						*dst++ = ap[j][l] + (*p1++ - *p2++);
					}

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
					if (i < DEBUGSTEPS) {
						debugfile << dist/MAX(l1,l2) << endl;
					}
					if (MAX(l1,l2) > 0.0f) {
						mdist = mdist + dist/MAX(l1,l2);
					}
				}
				if (i < DEBUGSTEPS) {
					debugfile.close();
				}

			}
			else if (algorithm == 2) {
				model->getKNN(ap[0], neighbours+extra_neighbours+1, nn_idx, dists);
				for (j = 0; j < seglength; j++) {
					p = cvMat(1,pcaembdim,MAT_TYPE, ap[j]);

					for (l = 0; l < neighbours; l++) {
						// Make sure none of the first neighbours is in the last N-seglength
						if (nn_idx[l] == ANN_NULL_IDX) break;
						a = 0;
						while ((uint)nn_idx[l] > N-seglength-1) {
							nn_idx[l] = nn_idx[neighbours+a++];
							if (a >= extra_neighbours) {
								// Uh oh, couldn't find a good neighbour.
								cerr << "Warning: Couldn't find enough good neighbours." << endl;
								nn_idx[l] = 0;
								break;
							}
						}
						p1 = (ANNcoord*)(nn[k]->data.ptr+l*nn[k]->step);
						p2 = (ANNcoord*)(nnn[k]->data.ptr+l*nnn[k]->step);
						p3 = model->getDataPoint(nn_idx[l]+j);
						p4 = model->getDataPoint(nn_idx[l]+j+1);
						for (a = 0; a < pcaembdim; a++) {
							*p1++ = *p3++;
							*p2++ = *p4++;
						}
					}
					if (l < neighbours) cerr << "Warning: Couldn't find enough neighbours." << endl;

					// Computes the mean of the nearest neighbours.
					cvReduce(nn[k], navg[k], 0, CV_REDUCE_AVG );

					// Computes the mean of the neigbours' successors
					cvReduce(nnn[k], navg_next[k], 0, CV_REDUCE_AVG );

					p1 = (ANNcoord*)navg_next[k]->data.ptr;
					p2 = (ANNcoord*)navg[k]->data.ptr;
					dst = (ANNcoord*)proj_next[k]->data.ptr;
					for (l = 0; l < pcaembdim; l++) {
						*dst++ = ap[j][l] + (*p1++ - *p2++);
					}

					// np is the subsequent point in the trajectory to be classified.
					np = cvMat(1, pcaembdim, MAT_TYPE, ap[j + 1]);

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
					if (MAX(l1,l2) > 0.0f) {
						mdist = mdist + dist/MAX(l1,l2);
					}
				}
			}
			else if (algorithm == 3) {
				// Try to reduce score variance by taking account distance from
				// line segments when constructing expected next points.
				for (j = 0; j < seglength; j++) {
					p = cvMat(1,pcaembdim,MAT_TYPE, ap[j]);
					model->getKNN(ap[j], neighbours+extra_neighbours+1, nn_idx, dists);

					for (l = 0; l < neighbours; l++) {
						// Make sure none of the first neighbours is N, 0, or invalid.
						// printf("Neighbour Before: %d\n", nn_idx[l]);
						if (nn_idx[l] == ANN_NULL_IDX) break;
						a = 0;
						while ((uint)nn_idx[l] >= N-3 || (uint)nn_idx[l] == 0) {
							nn_idx[l] = nn_idx[neighbours+a++];
							if (a >= extra_neighbours) {
								// Uh oh, couldn't find a good neighbour.
								cerr << "Warning: Couldn't find enough good neighbours." << endl;
								nn_idx[l] = 0;
								break;
							}
						}

						// Copy in the data.
						p1 = (ANNcoord*)(nn[k]->data.ptr+l*nn[k]->step);
						p2 = (ANNcoord*)(nnn[k]->data.ptr+l*nnn[k]->step);
						p3 = model->getDataPoint(nn_idx[l]);
						p4 = model->getDataPoint(nn_idx[l]+1);
						p5 = model->getDataPoint(nn_idx[l]+2);
						// printf("Neighbour %d/%d", (uint)nn_idx[l], N);
						// printf(": %x\n", p4);
						interpcoeff = get_interpolation_coefficient(ap[j], p3, p4, pcaembdim);
						if (interpcoeff < 0.0f) {
							// Back up one step.
							p5 = p4;
							p4 = p3;
							p3 = model->getDataPoint(nn_idx[l]-1);
							interpcoeff = get_interpolation_coefficient(ap[j], p3, p4, pcaembdim);
						}
						else if (interpcoeff > 1.0f) {
							// Move ahead one step
							p3 = p4;
							p4 = p5;
							p5 = model->getDataPoint(nn_idx[l]+3);
							interpcoeff = get_interpolation_coefficient(ap[j], p3, p4, pcaembdim);
						}
						if (interpcoeff < 0.0f) interpcoeff = 0.0f;
						if (interpcoeff > 1.0f) interpcoeff = 1.0f;
						for (a = 0; a < pcaembdim; a++) {
							*p1++ = (1.0f - interpcoeff) * *p3 + interpcoeff * *p4;
							*p2++ = (1.0f - interpcoeff) * *p4 + interpcoeff * *p5;
							p3++; p4++; p5++;
						}
					}
					if (l < neighbours) cout << "Warning: Couldn't find enough neighbours." << endl;

					// Computes the mean of the nearest neighbours.
					cvReduce(nn[k], navg[k], 0, CV_REDUCE_AVG );

					// Computes the mean of the neigbours' successors
					cvReduce(nnn[k], navg_next[k], 0, CV_REDUCE_AVG );

					p1 = (ANNcoord*)navg_next[k]->data.ptr;
					p2 = (ANNcoord*)navg[k]->data.ptr;
					dst = (ANNcoord*)proj_next[k]->data.ptr;
					for (l = 0; l < pcaembdim; l++) {
						*dst++ = ap[j][l] + (*p1++ - *p2++);
					}

					// np is the subsequent point in the trajectory to be classified.
					np = cvMat(1, pcaembdim, MAT_TYPE, ap[j + 1]);

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
					if (MAX(l1,l2) > 0.0f) {
						mdist = mdist + dist/MAX(l1,l2);
					}
				}
			}
			annDeallocPts(ap);
			// cvmSet(mdists, i, k, mdist);
			if (k > 0) cout << " ";
			cout << mdist;
		}
		cout << "\n";
		// Print progress bar
		if (verbosity > 0) {
			cerr << "[";
			progress = (uint)((float)i / (float)(length-seglength-1)* 50.0f);
			for (l = 0; l < 50; l++) {
				if (l <= progress)
					cerr << "#";
				else
					cerr << " ";
			}
			cerr << "]\r";
		}
	}
	if (verbosity > 0)
		cerr << "\n";
	//print_matrix(mdists);
	for (i = 0; i < M; i++) {
		cvReleaseMat(&navg[i]);
		cvReleaseMat(&navg_next[i]);
		cvReleaseMat(&nn[i]);
		cvReleaseMat(&nnn[i]);
	}
	cvReleaseMat(&mdists);
}

inline float get_interpolation_coefficient(ANNpoint p, ANNpoint p1, ANNpoint p2, uint dim) {
	uint i;
	float num = 0.0, denom = 0.0;
	ANNcoord *a1 = p1, *a2 = p2, *q = p;
	for (i = 0; i < dim; i++) {
		num = num + (*q - *a1) * (*a2 - *a1);
		denom = denom + (*a2 - *a1) * (*a2 - *a1);
		a1++; a2++; q++;
	}
	return num / denom;
}
