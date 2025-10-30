#include "funcs.h"
#include <fstream>
#include <random>

#include <iostream>

#include "Eigen/Core"
using namespace Eigen;
#include <thread>
#include<mutex>

//const double RANGE_OF_MAGNITUDE = 1000000000000000.0; //10^15 currently
const double RANGE_OF_MAGNITUDE = 1000;





double predict(const modelWeights& myModel, const dataPoint& myPoint)
{
    const int x_index = NUM_FEATURES; // get the dimension of x_i's
    double bias = myModel(0, x_index);// get the intercept

    // myModel.head(x_index) extracts the first x_index elements
    double feature_dot_product = myModel.head(x_index).dot(myPoint.head(x_index));

    return feature_dot_product + bias;
}

double cost_function(const modelWeights& myModel, const dataPoint& myPoint)
{
    double y_predicted = predict(myModel,myPoint);
    const int x_index = NUM_FEATURES; // get the dimension of x_i's
    double y_actual = myPoint(0,x_index+1);
    return ( std::pow(( y_actual - y_predicted),2) );
}


modelWeights calculate_gradient_one_point(const modelWeights& myModel, const dataPoint& myPoint)
{
    // The number of parameters (N) and the required length of the feature vector
    const int N_params = NUM_FEATURES + 1; 
    
    const int target_y_index = NUM_FEATURES + 1; 

    double y_predicted = predict(myModel, myPoint);
    double y_actual = myPoint(0, target_y_index); 
    double error = y_predicted - y_actual;

    // Calculate Gradient g_i = Error * x_i_extended
    // Correct Slicing: .head(N_params) extracts 0 to N_params - 1
    modelWeights gradient_vector = error * myPoint.head(N_params); 
    
    return gradient_vector;
}


modelWeights calculate_local_gradient(const modelWeights& myModel, const chunk& dataChunk)
{
    // N_params is the size of the feature vector and weight vector (N)
    const int N_params = NUM_FEATURES + 1;
    
    // B is the batch size (chunk.rows())
    const int B = dataChunk.rows();
    
    // Extract X_chunk [B x N]
    Eigen::MatrixXd X_chunk = dataChunk.leftCols(N_params);
    
    // Extract y_chunk [B x 1]
    Eigen::VectorXd y_chunk = dataChunk.rightCols(1); 
    
    // Convert row vector w (myModel) [1 x N] to column vector w^T (w_column) [N x 1]
    Eigen::VectorXd w_column = myModel.transpose();
    
    // Calculate Predictions: y_predicted = X * w (B x N) * (N x 1) = (B x 1)
    // This calculates w^T * x_i for all rows i: y_pred_i = w^T * x_i
    Eigen::VectorXd y_predicted = X_chunk * w_column;

    // Calculate Residuals (Error) Vector: E = y_predicted - y_actual (B x 1)
    Eigen::VectorXd residuals = y_predicted - y_chunk;

    // The matrix equivalent for the gradient sum is g_t^T = E^T * X_chunk
    
    // Transpose residuals to get E^T (1 x B)
    Eigen::RowVectorXd residuals_transposed = residuals.transpose(); 

    // Calculate Gradient Sum: g_t^T = E^T * X_chunk (1 x B) * (B x N) = (1 x N)
    // This is the SUM of [ (w^T * x_i - y_i) * x_i ] for all points in the batch.
    modelWeights gradient_row = residuals_transposed * X_chunk; 

    gradient_row = gradient_row / double(B);

    return gradient_row;
}

void update_weights(modelWeights& myModel, modelWeights& gradient, double rate)
{
    myModel += rate * (-1) *gradient;
    return;
}


//for sequential ver
void compute_chunk_graidents(modelWeights& myModel,fullDataset& myFullData,size_t size_of_chunk)
{
    const size_t total_rows = myFullData.rows();
    const size_t NChunks = total_rows / size_of_chunk;
    const size_t remainder = total_rows % size_of_chunk;

    // Process all full chunks
    for (size_t i = 0; i < NChunks; ++i)
    {
        auto dataChunk = myFullData.middleRows(i * size_of_chunk, size_of_chunk);
        modelWeights grad = calculate_local_gradient(myModel, dataChunk);

        update_weights(myModel, grad, ALPHA);
    }

    // Process remaining (partial) chunk if any
    if (remainder > 0)
    {
        auto lastChunk = myFullData.bottomRows(remainder);
        modelWeights grad = calculate_local_gradient(myModel, lastChunk);

        update_weights(myModel, grad, ALPHA);
    }

    return;
}


// the multithreded ver funcs:

void update_locked_weights( std::vector<modelWeights>& buffer,
                            std::mutex& m,
                            modelWeights& myModel, 
                            modelWeights& displacement, 
                            size_t n_threads
                        )
{

    //m.lock();
    std::lock_guard<std::mutex> lock(m);

    myModel += double((1.0/n_threads))*displacement;

    log_weights_into_buffer(buffer,myModel);

    //m.unlock();
}

//func each thread will be executing: (using locks)
void batch_compute( std::vector<modelWeights>& buffer,
                    std::mutex& m,
                    modelWeights& myModel,
                    fullDataset& myFullData,
                    size_t size_of_chunk,
                    size_t staleness_var,
                    size_t n_threads
                    )
{   
    std::cout << "Thread ID: " << std::this_thread::get_id() << std::endl;

    const size_t total_rows = myFullData.rows();
    const size_t NChunks = total_rows / size_of_chunk;
    const size_t remainder = total_rows % size_of_chunk;

    modelWeights displacement_calc;
    modelWeights initial_weights = myModel;
    modelWeights batch_weights = initial_weights;


    size_t count = 0;

    // Process all full chunks
    for (size_t i = 0; i < NChunks; ++i)
    {
        auto dataChunk = myFullData.middleRows(i * size_of_chunk, size_of_chunk);
        modelWeights grad = calculate_local_gradient(batch_weights, dataChunk);

        update_weights(batch_weights, grad, ALPHA);

        count++;

        if(count == staleness_var)
        {
            displacement_calc = batch_weights - initial_weights;
            update_locked_weights(buffer,m,myModel,displacement_calc,n_threads);
            batch_weights = myModel;
            initial_weights = batch_weights;
            count = 0;
        }
            

    }

    // Process remaining (partial) chunk if any
    if (remainder > 0)
    {
        auto lastChunk = myFullData.bottomRows(remainder);
        modelWeights grad = calculate_local_gradient(batch_weights, lastChunk);
        
        update_weights(batch_weights, grad, ALPHA);

        displacement_calc = batch_weights - initial_weights;
        
        update_locked_weights(buffer,m,myModel,displacement_calc,n_threads);
    }


}




















// IO, Utilities funcs. etc...

// Simple function w0 * x0 + b_intercept = y
// Generate synthetic 1D linear data with Gaussian noise
void generate_linear_data(const std::string& filename,
                          size_t N,
                          double true_w0,
                          double true_b,
                          double noise_std,
                          bool binary)
{
    std::mt19937_64 rng(42);  // fixed seed for reproducibility
    std::uniform_real_distribution<double> x_dist(-10.0, 10.0);
    std::normal_distribution<double> noise(0.0, noise_std);

    std::ofstream fout;
    if (binary)
        fout.open(filename, std::ios::binary);
    else
        fout.open(filename);

    if (!fout.is_open()) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return;
    }

    const size_t CHUNK_SIZE = 1'000'000; // generate 1M at a time
    std::vector<double> buffer;
    buffer.reserve(CHUNK_SIZE * 2); // (x, y)

    size_t written = 0;
    while (written < N) {
        buffer.clear();
        size_t n = std::min(CHUNK_SIZE, N - written);
        for (size_t i = 0; i < n; ++i) {
            double x0 = x_dist(rng);
            double y  = true_w0 * x0 + true_b + noise(rng);
            if (binary) {
                buffer.push_back(x0);
                buffer.push_back(y);
            } else {
                fout << x0 << " " << y << "\n";
            }
        }

        if (binary)
            fout.write(reinterpret_cast<char*>(buffer.data()),
                       buffer.size() * sizeof(double));

        written += n;
        std::cout << "\rGenerated " << written << " / " << N << " samples..." << std::flush;
    }

    fout.close();
    std::cout << "\nDone. Wrote " << N << " samples → " << filename << std::endl;
}


// Read the full binary dataset into Eigen matrix
bool load_full_dataset_binary(const std::string& filename, fullDataset& data)
{
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    if (!fin.is_open()) {
        std::cerr << "Error: cannot open file " << filename << std::endl;
        return false;
    }

    std::streamsize file_size = fin.tellg();
    fin.seekg(0, std::ios::beg);

    const size_t NUM_DOUBLES_PER_POINT = 2; // x0, y
    const size_t total_doubles = file_size / sizeof(double);
    const size_t num_points = total_doubles / NUM_DOUBLES_PER_POINT;

    std::vector<double> buffer(total_doubles);
    fin.read(reinterpret_cast<char*>(buffer.data()), file_size);
    fin.close();

    data.resize(num_points, NUM_FEATURES + 2); // rows × cols (x0, bias, y_actual)
    
    for (size_t i = 0; i < num_points; ++i) {
        data(i, 0) = buffer[i * 2 + 0]; // x₀
        data(i, 1) = 1.0;               // bias
        data(i, 2) = buffer[i * 2 + 1]; // y_actual
        

        //std::cout << "\rRead " << i << " / " << num_points << " points..." << std::flush;
    }

    std::cout << "Loaded " << num_points
              << " samples (" << (file_size / (1024.0 * 1024.0 * 1024.0))
              << " GB) into memory.\n";

    return true;
}


// the logging func:
void log_weights_into_buffer(std::vector<modelWeights>& buffer,
                             const modelWeights& model
                                )
{
    buffer.push_back(model);
}



//write to text file : 
void dump_weight_log_to_file(const std::string& filename,
                             const std::vector<modelWeights>& buffer)
{
    std::ofstream fout(filename);
    if (!fout.is_open()) {
        std::cerr << "Error opening " << filename << "\n";
        return;
    }

    for (const auto& w : buffer) {
        for (int i = 0; i < w.size(); ++i) {
            fout << w(i);
            if (i < w.size() - 1) fout << ",";
        }
        fout << "\n";
    }
}