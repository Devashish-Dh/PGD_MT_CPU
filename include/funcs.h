#include<vector>
#include <string>
#include "Eigen/Core"
using namespace Eigen;


#ifndef FUNCS_FILE // Check if the guard macro is *not* defined

#define FUNCS_FILE // Define the guard macro



const size_t NUM_FEATURES = 2;    // SLR case
//const size_t NUM_FEATURES = 20; // MLR case

//the learning rate alpha?
const double ALPHA = 0.1;

const size_t chunk_size = 512; // for batch gradient calculation...


//to try:
// a very simple func of w0 * x + B = y_pred
// the Rosenbrock function 
//a D dimensional func:

typedef Matrix<double, 1, (NUM_FEATURES+2)> dataPoint; // x_i's, then bias/intercept = 1, then y_actual (y_actual will be uninitialized in case of new data points / when predicting...)

typedef Matrix<double, 1, (NUM_FEATURES+1)> modelWeights; // w_i's and b_intercept

typedef Matrix<double, chunk_size, (NUM_FEATURES+2)> chunk; // +1 for the bais/intercept , +1 for the target

double predict(const modelWeights& myModel, const dataPoint& myPoint);

double cost_function(const modelWeights& myModel, const dataPoint& myPoint);

modelWeights calculate_gradient_one_point(const modelWeights& myModel, const dataPoint& myPoint);

modelWeights calculate_local_gradient(const modelWeights& myModel, const chunk& dataChunk);

modelWeights update_weights(modelWeights& myModel, modelWeights& gradient_from_chunk, double rate = ALPHA);

// ... content of the header file (declarations, etc.) ...




//IO funcs etc




#endif










