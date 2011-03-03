//============================================================================
// Name        : ManifoldExperiment.cpp
// Author      : 
// Version     :
// Copyright   : BSD
// Description : Constructs a time-delay embedding of the data, equipped with
//               a model of the dynamics using an approximation of the
//               one step transition matrix based on averaging the sample
//               dynamics of the k-nearest neighbours.
//============================================================================
/*
 * main.cpp
 *
 *  Created on: Jun 28, 2009
 *      Author: jordan
 */

#include <stdlib.h>
#include <cstring>
#include <ctype.h>
#include <iostream>
#include <fstream>
#include <ANN/ANN.h>
#include <Tisean/tsa.h>
#include <limits.h>

#include "Utils.h"
#include "BuildTree.h"
#include "TDEModel.h"

using namespace std;

#define WID_STR "Build a model"

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
    fprintf(stderr,"\t-m dimension [default: 2]\n");
    fprintf(stderr,"\t-p # dimension to reduce to using PCA [default: same as m, ie. no reduction]\n");
    fprintf(stderr,"\t-d delay [default: 1]\n");
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
    if ((out=check_option(str,n,'m','u')) != NULL) {
        sscanf(out,"%u",&settings->embdim);
        settings->embset=1;
    }
    if ((out=check_option(str,n,'p','u')) != NULL) {
        sscanf(out,"%u",&settings->pcaembdim);
        settings->pcaembset=1;
    }
    if ((out=check_option(str,n,'d','u')) != NULL) {
        sscanf(out,"%u",&settings->delay);
        settings->delayset=1;
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
	srand((unsigned)time(NULL));
	TDEModel *tdeModel;
        Settings settings = { ULONG_MAX, 0, 0xff, 1, 4, 32, 1, 2, 0, NULL, NULL, NULL, 0, 0, 0, 0, 1, 1 };
    char stin=0;
    // uint i, j;

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

    settings.pcaembset = settings.embdim > settings.pcaembdim;
    if (!settings.pcaembset) {
    	settings.pcaembdim = settings.embdim;
    }

    tdeModel = new TDEModel(&settings);
    tdeModel->DumpTree(settings.outfile);

    /*
    ANNpoint ap = tdeModel->getDataPoint(0);
    uint N = 1000;
    ANNpointArray pts = annAllocPts(N, settings.embdim);;
    tdeModel->simulateTrajectory(ap, pts, settings.embdim, N);
     */
    /*
    uint k = 8;
    ANNidxArray nn_idx;
    ANNdistArray dists;
    nn_idx = new ANNidx[k];
    dists = new ANNdist[k];

    tdeModel->getKNN(ap, k, nn_idx, dists);


    delete [] nn_idx;
    delete [] dists;
     */
    // DUMP Manifold and Trajectory
    /*
    ofstream fout("/tmp/trajectory.csv");
    for (i = 0; i < N; i++) {
    	fout << pts[i][0];
    	for (j = 1; j < settings.embdim; j++) {
    		fout << "\t";
    		fout << pts[i][j];
    	}
    	fout << endl;
    }
    fout.close();
    annDeallocPt(ap);
    delete [] pts;
	*/
    delete tdeModel;
    annClose();
    if (settings.column != NULL) free(settings.column);
    if (settings.infile != NULL) free(settings.infile);
    if (settings.outfile != NULL) free(settings.outfile);
    return 0;
}
