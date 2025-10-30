#include <fstream>
#include<vector>
#include <string>
#include<iostream>
#include "Eigen/Core"
using namespace Eigen;
#include <thread>
#include <mutex>


#ifndef FUNCS_FILE // Check if the guard macro is *not* defined

#define FUNCS_FILE // Define the guard macro



const size_t NUM_FEATURES = 1;    // SLR case
//const size_t NUM_FEATURES = 20; // MLR case

//the learning rate alpha?
const double ALPHA = 0.01;

const size_t chunk_size = 8192; // for batch gradient calculation...

const size_t SMT_chunk_size = 4096;


//to try:
// a very simple func of w0 * x + B = y_pred
// the Rosenbrock function 
//a D dimensional func:

typedef Matrix<double, 1, (NUM_FEATURES+2)> dataPoint; // x_i's, then bias/intercept = 1, then y_actual (y_actual will be uninitialized in case of new data points / IF predicting...)

typedef Matrix<double, 1, (NUM_FEATURES+1)> modelWeights; // w_i's and b_intercept

typedef Matrix<double, Dynamic, (NUM_FEATURES+2)> chunk; // +1 for the bais/intercept , +1 for the target

//for holding ALL THE DATA!
typedef Matrix<double, Dynamic, (NUM_FEATURES + 2)> fullDataset;


double predict(const modelWeights& myModel, const dataPoint& myPoint);

double cost_function(const modelWeights& myModel, const dataPoint& myPoint);

modelWeights calculate_gradient_one_point(const modelWeights& myModel, const dataPoint& myPoint);

modelWeights calculate_local_gradient(const modelWeights& myModel, const chunk& dataChunk);

void update_weights(modelWeights& myModel, modelWeights& gradient, double rate = ALPHA);

void compute_chunk_graidents(modelWeights& myModel,fullDataset& myFullData,size_t size_of_chunk = chunk_size);




//the updating the lock 
void update_locked_weights( std::vector<modelWeights>& buffer,
                            std::mutex& m,
                            modelWeights& myModel, 
                            modelWeights& displacement, 
                            size_t n_threads
                        );

// the batched GD
void batch_compute( std::vector<modelWeights>& buffer,
                    std::mutex& m,
                    modelWeights& myModel,
                    fullDataset& myFullData,
                    size_t size_of_chunk,
                    size_t staleness_var,
                    size_t n_threads
                    );




// ... content of the header file (declarations, etc.) ...




//IO funcs etc


// Generates synthetic linear data with Gaussian noise and saves to a text file
void generate_linear_data(const std::string& filename,
                                size_t N,
                                double true_w0,
                                double true_b,
                                double noise_std,
                                bool binary = true);

// Reads one chunk of data (x, y_actual) pairs from file into Eigen matrix
bool load_full_dataset_binary(const std::string& filename, fullDataset& data);

//the weights logging func
void log_weights_into_buffer(std::vector<modelWeights>& buffer,
                                const modelWeights& model
                                );



void dump_weight_log_to_file(const std::string& filename,
                             const std::vector<modelWeights>& buffer);





#endif










