/*
 * Classifier.h
 *
 *  Created on: 2009-06-30
 *      Author: jfrank8
 */

#include <stdlib.h>
#include "ClassifyTrajectory.h"
#include <vector>
#include <ANN/ANN.h>

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

class Classifier {

private:
	std::vector<NamedModel*> models;

public:
	Classifier(std::vector<NamedModel*> models);
	void go(ANNcoord* data, ulong length, ulong embdim, uint neighbours, uint seglength);
	virtual ~Classifier();
};

#endif /* CLASSIFIER_H_ */
