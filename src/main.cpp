#include <iostream>
#include "funcs.h"
#include "Eigen/Core"
#include <random>


//double : 64 bits == 8 Bytes

using namespace Eigen;
using namespace std;

const string BINARY_FILE_SLR = "SLR_1d_data.bin";

const size_t NUM_DATA_POINTS = 200000000;

int main()
{
  dataPoint mypt;
  mypt << 1,2,3,4;

  std::cout<<"sizeof( data_point ) = "<<sizeof(mypt)<<"bytes\n";

  // generate_linear_data(BINARY_FILE_SLR,NUM_DATA_POINTS,7.5,3.14,5,true);

  fullDataset myData;

  load_full_dataset_binary(BINARY_FILE_SLR, myData);

  std::cout<<"Dataset shape: " << myData.rows()<<" x " << myData.cols() << std::endl;



  return 1234;
}
