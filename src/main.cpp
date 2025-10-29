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
  mypt << 1,2,3;

  std::cout<<"sizeof( data_point ) = "<<sizeof(mypt)<<"bytes\n";

  // generate_linear_data(BINARY_FILE_SLR,NUM_DATA_POINTS,7.5,3.14,5,true);

  fullDataset myData;

  load_full_dataset_binary(BINARY_FILE_SLR, myData);

  std::cout<<"Dataset shape: " << myData.rows()<<" x " << myData.cols() << std::endl;

  // for(size_t row =0; row < 5; row++)
  // {
  //   for(size_t col = 0; col < myData.cols();col++)
  //   {
  //     std::cout<<myData(row,col)<<" ";
  //   }
  //   std::cout<<"\n";
  // }

  modelWeights MODEL = modelWeights::Zero();

  // for(size_t k = 0; k <10000; k++)
  // {
  //   modelWeights update = calculate_gradient_one_point(MODEL,myData.row(k));

  //   MODEL -= ALPHA*update;

  //   std::cout<<"w_0_calc: "<<MODEL(0,0)<<" b: "<<MODEL(0,1)<<"\n";
  // }



  
  //Pure_Sequential:
  compute_chunk_graidents(MODEL, myData);
  std::cout<<"w_0_calc: "<<MODEL(0,0)<<" b: "<<MODEL(0,1)<<"\n";


  //ONE THREAD PER CORE



  //SMT TWO THREADS PER CORE


  

  return 1234;
}
