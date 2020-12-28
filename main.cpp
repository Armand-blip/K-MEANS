#include <random>
#include <iostream>
#include <chrono>
#include <cstdio>
#include <omp.h>
#include "Header.h"

using namespace std;
using namespace chrono;

int main() {
    printf("K-means initialization\n");
    srand(time(NULL));
    // Dataset creation and filling, centroid randomization
    Centers centers;
  
    vector<float> a(Points, 0.0);
    vector<float> b(Points, 0.0);
    vector<int> group(Points, 0);

    srand(time(NULL));

  

    for (int i = 0; i < Points; i++) {
        a[i] = ((float)rand() / RAND_MAX) *::upper_bound;
        b[i] = ((float)rand() / RAND_MAX)*::upper_bound;
    }

    Datapoint* datapoint = new Datapoint(a, b, group);

    findCenters(*datapoint, &centers);

    for (int i = 0; i < centers_nr; i++) {
        printf("Starting position of the centroid number %i: x = %f, y = %f\n", i + 1, centers.a[i], centers.b[i]);
    }

    steady_clock::time_point t1 = steady_clock::now();
    // Compute K-Means on the given 2D points in the dataset.
    // Returns the number of points per cluster.
    // The parallel bool variable defines which function is called.
    if (parallel == true) {
        KMeans_parallel(*datapoint, &centers);
    }
    else {
        KMeans(*datapoint, &centers);
    }
    steady_clock::time_point t2 = steady_clock::now();

    printf("---------------------------------------------------------------------------\n");

    // Outputs the final position of the centroids
    for (int i = 0; i < centers_nr; i++) {
        printf("Final position of the centroid number %i: x = %f, y = %f\n", i + 1, centers.a[i], centers.b[i]);
    }

    if (parallel == true) { 
        printf("Using %d processors for computation. Total time: %d[ms]\n", omp_get_num_procs(), duration_cast<milliseconds>(t2 - t1).count());
    }
    else {
        printf("Running on single thread. Total time: %d[ms]\n", duration_cast<milliseconds>(t2 - t1).count());
    }

    return 0;
}
