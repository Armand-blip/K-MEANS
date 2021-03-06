#pragma once
// This header is created for declaration and initalization of variables and functions.
#include <vector>
using namespace std;

// Initialization of variables
const int Points = 10000;
const int centers_nr = 20;
const float upper_bound = 1000.0;
const int Iterations = 10;
const bool parallel = true;

// We create a class for defining the datapoints sizes and dimensions
// Defining a,b and group, group is the cluster in which the points correspond to.
class Datapoint {
private:
    vector<float> a;
    vector<float> b;
    vector<int> group;
public:
    Datapoint(vector<float> a, vector<float> b, vector<int> group) {
        this->a = a;    // The 'this' pointer is used to retrieve the object's a hidden by the private 'a' 
        this->b = b;    
        this->group = group;
    }
    // We use the get and set method to access private attributes
    float getA(int ind) {
        return this->a[ind];  //returns the current object instance
    }
    float getB(int ind) {
        return this->b[ind];
    }
    int getgroup(int ind) {
        return this->group[ind];
    }
    //and then the setters
    void setGroup(int i, int group) {
        this->group[i] = group;
    }
};
// Since we need more than one variable in order to represent an object, the best way is to create a struct.
struct Centers {
    float a[centers_nr];
    float b[centers_nr];
    float a_new[centers_nr];
    float b_new[centers_nr];
    int point_nr[centers_nr];
};

// Then, we need to declare the functions
float EuclideanDist(Datapoint& datapoint, Centers* centers, int indPoint, int indCenter);
void findCenters(Datapoint& datapoint, Centers* centers);
void KMeans(Datapoint& datapoint, Centers* centers);
void KMeans_parallel(Datapoint& datapoint, Centers* centers);
void modifycenter_new(Centers* centers, int indCenter);
void groupPoint(Datapoint& datapoint, Centers* centers, int indPoint);
