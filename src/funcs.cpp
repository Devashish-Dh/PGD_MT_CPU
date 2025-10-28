#include "funcs.h"

#include <iostream>
#include "Eigen/Core"
using namespace Eigen;

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

modelWeights update_weights(modelWeights& myModel, modelWeights& gradient_from_chunk, double rate)
{
    myModel += rate * (-1) *gradient_from_chunk;
    return myModel;
}



// IO funcs etc