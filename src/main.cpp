#include <iostream>
#include "funcs.h"
#include "Eigen/Core"
#include <random>
#include <mutex>
#include <thread>
#include <vector>
//double : 64 bits == 8 Bytes

using namespace Eigen;
using namespace std;

const string BINARY_FILE_SLR = "SLR_1d_data.bin";

const size_t STALENESS_VAR = 1024;

const size_t NUM_DATA_POINTS = 200000000;


const string UPDATE_GLOBAL_WEIGHTS = "weight_updates.txt";

int main()
{
  // dataPoint mypt;
  // mypt << 1,2,3;

  // std::cout<<"sizeof( data_point ) = "<<sizeof(mypt)<<"bytes\n";

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
  // compute_chunk_graidents(MODEL, myData);
  // std::cout<<"w_0_calc: "<<MODEL(0,0)<<" b: "<<MODEL(0,1)<<"\n";

  std::vector<modelWeights> weight_snapshots;



  unsigned int n_max = std::thread::hardware_concurrency();
  std::cout << "Number of concurrent threads supported by hardware: " << n_max << std::endl;

  std::mutex m_for_weights;
  

  //SMT TWO THREADS PER CORE (using locks)


  //code to spawn the threads and call batch_compute in each for a disjoint piece of the data : 

 // Partition data rows per thread
    const size_t total_rows = myData.rows();
    const size_t rows_per_thread = static_cast<size_t>(
        std::ceil(static_cast<double>(total_rows) / n_max)
    );

    std::vector<std::thread> workers;
    workers.reserve(n_max);

    for (size_t t = 0; t < n_max; ++t) {
        size_t start = t * rows_per_thread;
        if (start >= total_rows) break;

        size_t end = std::min(start + rows_per_thread, total_rows);
        size_t rows_this_thread = end - start;

        // Each thread gets its disjoint row block (read-only)
        // Use .middleRows() to make sure thread gets its own copy
        fullDataset dataSlice = myData.middleRows(start, rows_this_thread);
   
        workers.emplace_back([&, dataSlice]() mutable {
    batch_compute(weight_snapshots,m_for_weights, MODEL, dataSlice,
                  SMT_chunk_size, STALENESS_VAR, n_max);});
    }

    // join threads
    for (auto& th : workers) {
        if (th.joinable()) th.join();
    }

    std::cout << "All threads finished.\n";
  
    std::cout<<"w_0_calc: "<<MODEL(0,0)<<" b: "<<MODEL(0,1)<<"\n";

    dump_weight_log_to_file(UPDATE_GLOBAL_WEIGHTS, weight_snapshots);

    std::cout << "dumped weight updates to text file!\n";




  return 0;
}
