#include <time.h> //This header file contains definitions of functions to get and manipulate date and time information.
#include <cmath>
#include <limits>
#include <algorithm>
#include <vector>
#include "Header.h"
#include <fstream>

using namespace std;

// Function that pick points randomly from datapoints and make it the first center  
void findCenters(Datapoint &datapoint, Centers *centers) {
    srand(time(NULL));    //seeds the pseudo random number generator that rand() uses
    for (int i = 0; i < centers_nr; i++) {
        int ind_center = rand() % Points;
        centers->a[i] = datapoint.getA(ind_center);
        centers->b[i] = datapoint.getB(ind_center);
        centers->a_new[i] = 0;
        centers->b_new[i] = 0;
        centers->point_nr[i] = 0;
    }
};

// A function that calculates the distance between a point and the centroid of the group according to Euclidean Method
float EuclideanDist(Datapoint &datapoint, Centers *centers, int indPoint, int indCenter) {
    return sqrt(pow(datapoint.getA(indPoint) - centers->a[indCenter], 2) + pow(datapoint.getB(indPoint) - centers->b[indCenter], 2));
}
// Function that resets the coordinate accumulators of a centroid
void modifycenter_new(Centers *centers, int indCenter) {
    centers->a_new[indCenter] = 0;
    centers->b_new[indCenter] = 0;
    centers->point_nr[indCenter] = 0;
}

// Function that adds a point to a cluster using OpenMP atomic directive to make each operation atomic
void groupPoint(Datapoint &datapoint, Centers *centers, int indPoint) {
    centers->a_new[datapoint.getgroup(indPoint)] += datapoint.getA(indPoint);
    centers->b_new[datapoint.getgroup(indPoint)] += datapoint.getB(indPoint);
    centers->point_nr[datapoint.getgroup(indPoint)] += 1;
}
// K-Means function.Computes K-Means from a 2D dataset with previously chosen centroids.
// It progressively modifies the coordinates of the centroids and stops when the problem converges.
void KMeans(Datapoint &datapoint, Centers *centers) {

    for (int i = 0; i < Iterations; i++); {

        for (int j = 0; j < Points; j++) {
            int nearest_center = 0;
            float smallest_dist = numeric_limits<float>::max();

            for (int k = 0; k < centers_nr; k++) {
                float current_dist = EuclideanDist(datapoint, centers, j, k);
                if (current_dist < smallest_dist) {
                    smallest_dist = current_dist;
                    nearest_center = k;
                }
            }

            datapoint.setGroup(j, nearest_center);
            groupPoint(datapoint, centers, j);
        }
        // Computing the new centroid coordinates
        for (int k = 0; k < centers_nr; k++) {
            if (centers->a_new[k] != 0) {
                centers->a[k] = centers->a_new[k] / std::max(centers->point_nr[k], 1);
                centers->b[k] = centers->b_new[k] / std::max(centers->point_nr[k], 1);
            }
            modifycenter_new(centers, k);
        }

    }
};
// Parallel version of the function.
void KMeans_parallel(Datapoint &datapoint, Centers *centers) {
    for (int i = 0; i < Iterations; i++) {
        int nearest_center;
        float smallest_dist;
        Centers local_centroids;
#pragma omp parallel default(shared) private(smallest_dist, nearest_center, local_centroids) // num_threads(4)
        {
            local_centroids = *centers;
#pragma omp for schedule(static)
            for (int j = 0; j < Points; ++j) {
                nearest_center = 0;
                smallest_dist = numeric_limits<float>::max(); //Returns the maximum finite value representable by the numeric type float.(FLT_MAX)
                for (int k = 0; k < centers_nr; ++k) {
                    float current_dist = EuclideanDist(datapoint, centers, j, k);
                    // Assigning the j-th point to the closest cluster
                    if (current_dist < smallest_dist) {
                        smallest_dist = current_dist;
                        nearest_center = k;
                    }
                }
                datapoint.setGroup(j, nearest_center);
                local_centroids.a_new[nearest_center] += datapoint.getA(j);
                local_centroids.b_new[nearest_center] += datapoint.getB(j);
                ++local_centroids.point_nr[nearest_center];
            }
#pragma omp critical
            for (int k = 0; k < centers_nr; ++k) {
                centers->a_new[k] += local_centroids.a_new[k];
                centers->b_new[k] += local_centroids.b_new[k];
                centers->point_nr[k] += local_centroids.point_nr[k];
            }
        }
        // Computing the new centroid coordinates
        for (int k = 0; k < centers_nr; k++) {
            if (centers->a_new[k] != 0) {
                centers->a[k] = centers->a_new[k] / std::max(centers->point_nr[k], 1);
                centers->b[k] = centers->b_new[k] / std::max(centers->point_nr[k], 1);
            }
            modifycenter_new(centers, k);
        }
    }
}

