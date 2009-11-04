/*
 * ClassifyTrajectory.cpp
 *
 *  Created on: 2009-06-30
 *      Author: jfrank8
 */

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
    fprintf(stderr,"\nUsage: %s [options]\n",progname);
    fprintf(stderr,"Options:\n");
    fprintf(stderr,"Everything not being a valid option will be interpreted as a"
            " possible datafile.\nIf no datafile is given stdin is read."
            " Just - also means stdin\n");
    fprintf(stderr,"\t-l # of data [default: whole file]\n");
    fprintf(stderr,"\t-x # of rows to ignore [default: 0]\n");
    fprintf(stderr,"\t-M num. of columns to read [default: 1]\n");
    fprintf(stderr,"\t-c columns to read [default: 1,...,M]\n");
    fprintf(stderr,"\t-V verbosity level [default: 1]\n\t\t"
            "0='only panic messages'\n\t\t"
            "1='+ input/output messages'\n");
    fprintf(stderr,"\t-o output file [default: 'datafile'.del, "
            "without -o: stdout]\n");
    fprintf(stderr,"\t-h show these options\n");
    exit(0);
}
void scan_options(int n,char **str, Settings *settings) {
    char *out;
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
    if ((out=check_option(str,n,'V','u')) != NULL)
        sscanf(out,"%u",&settings->verbosity);
    if ((out=check_option(str,n,'o','o')) != NULL) {
        settings->stdo=0;
        if (strlen(out) > 0)
            settings->outfile=out;
    }
}

int main (int argc, char *argv[]) {
	vector<NamedModel*> models;
	NamedModel *model;
	Classifier *classifier;
    Settings settings = { ULONG_MAX, 0, 0xff, 1, 1, 2, NULL, NULL, NULL, 0, 0, 0, 1 };
    double** data; // one double* for each model.
    double *proj;
    ulong *tlengths, tlength;
    char stin=0;
    char *model_ini = (char*)"models.ini";
    ifstream models_file;
    char buf[50];
    int n = 50;
    uint i;

    if (scan_help(argc,argv))
        show_options(argv[0]);
    scan_options(argc,argv, &settings);

    settings.infile=search_datafile(argc,argv,NULL,settings.verbosity);
    if (settings.infile == NULL)
        stin=1;
    if (settings.outfile == NULL) {
        if (!stin) {
            check_alloc(settings.outfile=(char*)calloc(strlen(settings.infile)+5,sizeof(char)));
            strcpy(settings.outfile,settings.infile);
            strcat(settings.outfile,".dmp");
        } else {
            check_alloc(settings.outfile=(char*)calloc(10,sizeof(char)));
            strcpy(settings.outfile,"stdin.dmp");
        }
    }
    if (!settings.stdo) {
        test_outfile(settings.outfile);
    }

    if (settings.delay < 1) {
        fprintf(stderr,"Delay has to be larger than 0. Exiting!\n");
        exit(DELAY_SMALL_ZERO);
    }

    // Read the models.ini file.
    models_file.open(model_ini);
    while(!models_file.eof()) {
    	models_file.getline(buf, n);
    	if (strlen(buf) > 0) {
    		check_alloc(model=(NamedModel*)calloc(1,sizeof(NamedModel)));
    		check_alloc(model->name=(char*)calloc(strlen(buf)+1,sizeof(char)));
    		strcpy(model->name,buf);
    		model->model = new TDEModel(new ifstream(model->name));
    		models.push_back(model);
    	}
    }
    data = new double*[models.size()];
    tlengths = new ulong[models.size()];
    for (i = 0; i < models.size(); i++) {
        settings.embset = 1;
        settings.embdim = models[i]->model->getEmbDim();
        settings.delayset = 1;
        settings.delay = models[i]->model->getDelay();

        get_embedding(&settings, data[i], tlengths[i]);

        settings.pcaembset = models[i]->model->getUsePCA();
        if (settings.pcaembset) {
        	settings.pcaembdim = models[i]->model->getPCAEmbDim();
        	proj = models[i]->model->projectData(data[i],tlengths[i],settings.embdim);
        	delete [] data[i];
        	data[i] = proj;
        }
        else {
        	settings.pcaembdim = settings.embdim;
        }
    }
    // Since the projected data might have different lengths under different models,
    // we just use the minimum so that all points are compared under all models.
    tlength = tlengths[0];
    for (i = 1; i < models.size(); i++) {
    	if (tlengths[i] < tlength) tlength = tlengths[i];
    }

    classifier = new Classifier(models);
    classifier->go(data, tlength);

    for (i = 0; i < models.size(); i++) {
    	free(models[i]->name);
    	delete models[i]->model;
    	free(models[i]);
    	delete [] data[i];
    }
    delete [] data;
    delete [] tlengths;

    annClose();
    if (settings.column != NULL) free(settings.column);
    if (settings.infile != NULL) free(settings.infile);
    if (settings.outfile != NULL) free(settings.outfile);
}
