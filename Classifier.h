//============================================================================
// Name        : Classifier.h
// Author      : Jordan Frank (jordan.frank@cs.mcgill.ca)
// Copyright   : MIT
//============================================================================

#include <stdlib.h>
#include "ClassifyTrajectory.h"
#include <vector>
#include <ANN/ANN.h>

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

class Classifier {

private:
	std::vector<NamedModel*>* models;

public:
	Classifier(std::vector<NamedModel*>* models);
	void go(ANNcoord* data, uint length, uint embdim, uint neighbours, uint seglength, uint algorithm, uint verbosity);
	virtual ~Classifier();
};

inline float get_interpolation_coefficient(ANNpoint p, ANNpoint p1, ANNpoint p2, uint dim);
#endif /* CLASSIFIER_H_ */
