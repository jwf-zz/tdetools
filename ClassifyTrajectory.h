//============================================================================
// Name        : ClassifyTrajectory.h
// Author      : Jordan Frank (jordan.frank@cs.mcgill.ca)
// Copyright   : MIT
//============================================================================

#include "TDEModel.h"

#ifndef CLASSIFYTRAJECTORY_H_
#define CLASSIFYTRAJECTORY_H_

typedef struct {
	TDEModel* model;
	char* name;
} NamedModel;

class ClassifyTrajectory {
public:
	ClassifyTrajectory();
	virtual ~ClassifyTrajectory();
};

#endif /* CLASSIFYTRAJECTORY_H_ */
