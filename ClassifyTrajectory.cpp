//============================================================================
// Name        : ClassifyTrajectory.cpp
// Author      : Jordan Frank (jordan.frank@cs.mcgill.ca)
// Copyright   : MIT
// Description : Application for running the GTM classifier on a data set.
//============================================================================

#include <stdlib.h>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <ANN/ANN.h>
#include <Tisean/tsa.h>

#include "Utils.h"
#include "BuildTree.h"
#include "TDEModel.h"
#include "ClassifyTrajectory.h"
#include "Classifier.h"

using namespace std;

using namespace std;

#define WID_STR "Classify a set of trajectories"

void show_options(char *progname) {
    what_i_do(progname, (char*)WID_STR);
    fprintf(stderr,"\nUsage: %s [options] [datafile]\n",progname);
    fprintf(stderr,"Options:\n");
    fprintf(stderr,"If no datafile is given stdin is read."
            " Just - also means stdin\n");
    fprintf(stderr,"\t-l # of data [default: whole file]\n");
    fprintf(stderr,"\t-x # of rows to ignore [default: 0]\n");
    fprintf(stderr,"\t-M num. of columns to read [default: 1]\n");
    fprintf(stderr,"\t-c columns to read [default: 1]\n");
    fprintf(stderr,"\t-n number of neighbours to match [default: 4]\n");
    fprintf(stderr,"\t-s length of matched segments [default: 32]\n");
    fprintf(stderr,"\t-A # of algorithm to use (1: old, 2: new (unpublished), 3: fancy new (still buggy)) [default: old algorithm (1)]\n");
    fprintf(stderr,"\t-V verbosity level [default: 0]\n\t\t"
            "0='only panic messages'\n\t\t"
            "1='+ input/output messages'\n");
    fprintf(stderr,"\t-h show these options\n");
    exit(0);
}
void scan_options(int n,char **str, Settings *settings) {
    char *out;
    if ((out=check_option(str,n,'A','u')) != NULL)
        sscanf(out,"%u",&settings->algorithm);
    if ((out=check_option(str,n,'l','u')) != NULL)
        sscanf(out,"%lu",&settings->length);
    if ((out=check_option(str,n,'x','u')) != NULL)
        sscanf(out,"%lu",&settings->exclude);
    if ((out=check_option(str,n,'c','s')) != NULL)
        settings->column=out;
    if ((out=check_option(str,n,'M','u')) != NULL) {
        sscanf(out,"%u",&settings->indim);
        settings->dimset=1;
    }
    if ((out=check_option(str,n,'n','u')) != NULL) {
        sscanf(out,"%u",&settings->neighbours);
    }
    if ((out=check_option(str,n,'s','u')) != NULL) {
        sscanf(out,"%u",&settings->seglength);
    }
    if ((out=check_option(str,n,'V','u')) != NULL)
        sscanf(out,"%u",&settings->verbosity);
}

int main (int argc, char *argv[]) {
	vector<NamedModel*> models;
	NamedModel *model;
	Classifier *classifier;
	Settings settings = { ULONG_MAX, 0, 0, 3, 4, 32, 1, 5, 5, NULL, NULL, NULL, 0, 0, 0, 0, 1, 1 };
    ANNcoord* data;
    unsigned long tlength;
    char stin=0;
    char *model_ini = (char*)"models.ini";
    ifstream models_file;
    char buf[500];
    int n = 500;

    if (scan_help(argc,argv))
        show_options(argv[0]);
    scan_options(argc,argv, &settings);

    settings.infile=search_datafile(argc,argv,NULL,settings.verbosity);
    if (settings.infile == NULL)
        stin=1;

    if (settings.delay < 1) {
        fprintf(stderr,"Delay has to be larger than 0. Exiting!\n");
        exit(DELAY_SMALL_ZERO);
    }
    if (settings.algorithm < 1 || settings.algorithm > 3) {
    	fprintf(stderr, "Invalid algorithm: Value must be 1, 2, or 3.\n");
    	exit(1);
    }

    // Read the models.ini file.
    models_file.open(model_ini);
    while(!models_file.eof()) {
    	models_file.getline(buf, n);
    	if (strlen(buf) > 0) {
    		check_alloc(model=(NamedModel*)calloc(1,sizeof(NamedModel)));
    		check_alloc(model->name=(char*)calloc(strlen(buf)+1,sizeof(char)));
    		strcpy(model->name,buf);
    		if (settings.verbosity > 0)
    			cerr << "Reading from model: " << model->name << endl;
    		model->model = new TDEModel(new ifstream(model->name), settings.verbosity);
    		models.push_back(model);
    	}
    }
    // Get delay embedding settings from one of the models
    settings.embset = 1;
    settings.embdim = models[0]->model->getEmbDim();
    settings.delayset = 1;
    settings.delay = models[0]->model->getDelay();
    settings.pcaembset = models[0]->model->getUsePCA();
    if (settings.pcaembset) {
    	settings.pcaembdim = models[0]->model->getPCAEmbDim();
    }
    else {
    	settings.pcaembdim = settings.embdim;
    }

    get_embedding(&settings, data, tlength);

    cerr << "Classifying..." << endl;
    classifier = new Classifier(&models);
    classifier->go(data, tlength, settings.embdim, settings.neighbours, settings.seglength, settings.algorithm, settings.verbosity);

    for (uint i = 0; i < models.size(); i++) {
    	free(models[i]->name);
    	delete models[i]->model;
    	free(models[i]);
    }
    free(data);
    delete classifier;
    annClose();
    if (settings.column != NULL) free(settings.column);
    if (settings.infile != NULL) free(settings.infile);
    if (settings.outfile != NULL) free(settings.outfile);
}
