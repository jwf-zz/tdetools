//============================================================================
// Name        : TDEModel.h
// Author      : Jordan Frank (jordan.frank@cs.mcgill.ca)
// Copyright   : MIT
//============================================================================

#include <opencv/cxcore.h>
#include "Utils.h"

#ifndef TDEMODEL_H_
#define TDEMODEL_H_

class TDEModel {
public:
	TDEModel(Settings* settings);
	TDEModel(std::ifstream* model_file, uint verbosity);
	virtual ~TDEModel();

	void DumpTree(char* outfile);
	void getKNN(ANNpoint ap, unsigned int k, ANNidxArray nn_idx, ANNdistArray dists);
	void simulateTrajectory(ANNpoint s0, ANNpointArray trajectory, unsigned int dim, unsigned long  N);
    ANNpoint getDataPoint(unsigned int idx);
    ANNcoord *projectData(ANNcoord *data, unsigned int rows, unsigned int cols);

    unsigned long getLength() const { return length; }
    unsigned long getEmbDim() const { return embdim; }
    unsigned long getDelay() const { return delay; }
    char getUsePCA() const { return use_pca; }
    unsigned long getPCAEmbDim() const {
    	if (use_pca) {
    		return bases->cols;
    	}
    	else {
    		return embdim;
    	}
    }
private:
    unsigned long length, embdim, delay;
    ANNpointArray dataPts;
    ANNkd_tree *kdTree;
    // Related to the PCA
    void computePCABases(ANNcoord *data, uint rows, uint cols, uint numbases);
    char use_pca;
    CvMat* avg;
    CvMat* bases;
};

#endif /* TDEMODEL_H_ */
