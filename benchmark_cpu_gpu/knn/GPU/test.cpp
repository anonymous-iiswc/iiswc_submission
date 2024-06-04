#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <helper_string.h>  // helper for shared functions common to CUDA Samples

#include "knncuda.h"
//------------------
#include <VaryTypeAndOperator.h>
#include <PrintOutput.h>

MultiLineExternalVariablesMacros


/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    srand(time(NULL));

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }
}


/**
 * Computes the Euclidean distance between a reference point and a query point.
 *
 * @param ref          refence points
 * @param ref_nb       number of reference points
 * @param query        query points
 * @param query_nb     number of query points
 * @param dim          dimension of points
 * @param ref_index    index to the reference point to consider
 * @param query_index  index to the query point to consider
 * @return computed distance
 */
float compute_distance(const float * ref,
                       int           ref_nb,
                       const float * query,
                       int           query_nb,
                       int           dim,
                       int           ref_index,
                       int           query_index) {
    float sum = 0.f;
    for (int d=0; d<dim; ++d) {
        const float diff = ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
        sum += abs(diff);
    }
    return sum;
}


/**
 * Gathers at the beginning of the `dist` array the k smallest values and their
 * respective index (in the initial array) in the `index` array. After this call,
 * only the k-smallest distances are available. All other distances might be lost.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist    array containing the `length` distances
 * @param index   array containing the index of the k smallest distances
 * @param length  total number of distances
 * @param k       number of smallest distances to locate
 */
void  modified_insertion_sort(float *dist, int *index, int length, int k){

    // Initialise the first index
    index[0] = 0;

    // Go through all points
    for (int i=1; i<length; ++i) {

        // Store current distance and associated index
        float curr_dist  = dist[i];
        int   curr_index = i;

        // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
        if (i >= k && curr_dist >= dist[k-1]) {
            continue;
        }

        // Shift values (and indexes) higher that the current distance to the right
        int j = std::min(i, k-1);
        while (j > 0 && dist[j-1] > curr_dist) {
            dist[j]  = dist[j-1];
            index[j] = index[j-1];
            --j;
        }

        // Write the current distance and index at their position
        dist[j]  = curr_dist;
        index[j] = curr_index; 
    }
}


/*
 * For each input query point, locates the k-NN (indexes and distances) among the reference points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 * @param k          number of neighbors to consider
 * @param knn_dist   output array containing the query_nb x k distances
 * @param knn_index  output array containing the query_nb x k indexes
 */
bool knn_c(const float * ref,
           int           ref_nb,
           const float * query,
           int           query_nb,
           int           dim,
           int           k,
           float *       knn_dist,
           int *         knn_index) {

    // Allocate local array to store all the distances / indexes for a given query point 
    float * dist  = (float *) malloc(ref_nb * sizeof(float));
    int *   index = (int *)   malloc(ref_nb * sizeof(int));

    // Allocation checks
    if (!dist || !index) {
        printf("Memory allocation error\n");
        free(dist);
        free(index);
        return false;
    }

    // Process one query point at the time
    for (int i=0; i<query_nb; ++i) {

        // Compute all distances / indexes
        for (int j=0; j<ref_nb; ++j) {
            dist[j]  = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
            index[j] = j;
        }

        // Sort distances / indexes
        modified_insertion_sort(dist, index, ref_nb, k);

        // Copy k smallest distances and their associated index
        for (int j=0; j<k; ++j) {
            knn_dist[j * query_nb + i]  = dist[j];
            knn_index[j * query_nb + i] = index[j];
        }
    }

    // Memory clean-up
    free(dist);
    free(index);

    return true;

}


/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 * The tested function is ran several times in order to have a better estimate
 * of the processing time.
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param gt_knn_dist    ground truth distances
 * @param gt_knn_index   ground truth indexes
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * @param nb_iterations  number of iterations
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *, int, double &),
          const char *  name,
          int           nb_iterations, double &elapsed_time, bool gold_test) {

    // Parameters
    const float precision    = 0.001f; // distance error max
    const float min_accuracy = 0.999f; // percentage of correct values required

    // Display k-NN function name
    printf("- %-17s : ", name);

    // Allocate memory for computed k-NN neighbors
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_dist);
        free(test_knn_index);
        return false;
    }
    // Start timer
#ifndef ONLY_TIME_KERNELS
    struct timeval tic;
    gettimeofday(&tic, NULL);
#endif
    // Compute k-NN several times
#ifndef ONLY_TIME_KERNELS
    for (int i=0; i<nb_iterations; ++i) {
#endif
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index, nb_iterations, elapsed_time)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
#ifndef ONLY_TIME_KERNELS
    }
#endif
#ifndef ONLY_TIME_KERNELS
    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;
#endif
    // Verify both precisions and indexes of the k-NN values
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i=0; i<query_nb*k; ++i) {
        if (fabs(test_knn_dist[i] - gt_knn_dist[i]) <= precision) {
            nb_correct_precisions++;
        }
        if (test_knn_index[i] == gt_knn_index[i]) {
            nb_correct_indexes++;
        }
    }

    // Compute accuracy
    float precision_accuracy = nb_correct_precisions / ((float) query_nb * k);
    float index_accuracy     = nb_correct_indexes    / ((float) query_nb * k);

    // Display report
    if(gold_test)
    {
		if (precision_accuracy >= min_accuracy && index_accuracy >= min_accuracy ) {
			printf("PASSED in %8.5f seconds (averaged over %3d iterations)\n", elapsed_time / nb_iterations, nb_iterations);
		}
		else {
			printf("FAILED\n");
		}
    }else{
    		printf("Not Tested in %8.5f seconds (averaged over %3d iterations)\n", elapsed_time / nb_iterations, nb_iterations);
    }
    timeInMsec=(elapsed_time / nb_iterations)*1000.0f;

    sizeInGBytes= (ref_nb   * dim * sizeof(float)+ query_nb * dim * sizeof(float)  )* 1.0e-9;
    outPutSizeInGBytes=(query_nb * k   * sizeof(float)+query_nb * k   * sizeof(int))*1.0e-9;
    if(timeInMsec!=0){
    	  gigaProcessedInSec=( sizeInGBytes) / (timeInMsec / 1000.0f);
   }
    printOutput();
    // Free memory
    free(test_knn_dist);
    free(test_knn_index);

    return true;
}


/**
 * 1. Create the synthetic data (reference and query points).
 * 2. Compute the ground truth.
 * 3. Test the different implementation of the k-NN algorithm.
 */
#ifndef METRIC_RUN_MAIN
int main(int argc, char **argv)
#else
bool runPass(int argc, char **argv)
#endif
{

    // Parameters
    int ref_nb   = 16384;
    int query_nb = 4096;
    int dim      = 128;
    int k        = 16;


    bool global_mem = checkCmdLineFlag(argc, (const char **)argv, "global_mem");
    bool txt_mem = checkCmdLineFlag(argc, (const char **)argv, "txt_mem");
    bool cublas = checkCmdLineFlag(argc, (const char **)argv, "cublas");
    bool gold_test = checkCmdLineFlag(argc, (const char **)argv, "gold_test");

    if (checkCmdLineFlag(argc, (const char **)argv, "nIter"))
    {
    		nIter = getCmdLineArgumentInt(argc, (const char **)argv, "nIter");
    }else{
		#ifndef METRIC_RUN_ONLY_ONCE
    		nIter=30;
		#else
			nIter = 1;
		#endif
	}
    int numberOfIteration=nIter;
    noWarmUp=checkCmdLineFlag(argc, (const char **)argv, "NoWarmUp");
    if (checkCmdLineFlag(argc, (const char **)argv, "ref_nb"))
    {
    		ref_nb= getCmdLineArgumentInt(argc, (const char **)argv, "ref_nb");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "k"))
    {
    		k= getCmdLineArgumentInt(argc, (const char **)argv, "k");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "dim"))
    {
    		dim= getCmdLineArgumentInt(argc, (const char **)argv, "dim");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "query_nb"))
    {
    		query_nb= getCmdLineArgumentInt(argc, (const char **)argv, "query_nb");
    }
    // Display

    printf("PARAMETERS\n");
         printf("- Number reference points : %d\n",   ref_nb);
         printf("- Number query points     : %d\n",   query_nb);
         printf("- Dimension of points     : %d\n",   dim);
         printf("- Number of neighbors     : %d\n\n", k);
    // Sanity check
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }
    double elapsed_time;
    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);

    // Compute the ground truth k-NN distances and indexes for each query point
    if(gold_test){
		printf("Ground truth computation in progress...\n\n");
		if (!knn_c(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
			free(ref);
			free(query);
			free(knn_dist);
			free(knn_index);
			return EXIT_FAILURE;
		}
    }


    // Test all k-NN functions
    printf("TESTS\n");
    //test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_c,            "knn_c",              2);
    if(global_mem){
    	test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_global,  "knn_cuda_global",  numberOfIteration, elapsed_time, gold_test);
    }
    if(txt_mem){
    	test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_texture, "knn_cuda_texture", numberOfIteration,elapsed_time,gold_test);
    }
    if(cublas){
    	test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cublas,       "knn_cublas",       numberOfIteration,elapsed_time,gold_test);
    }
    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);
#ifndef METRIC_RUN_MAIN
    return EXIT_SUCCESS;
#else
    return true;
#endif

}
