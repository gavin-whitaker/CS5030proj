// mpi_cuda/main.cu — Entry point for the MPI+CUDA hybrid K-Means implementation
// TODO: initialize MPI, distribute data, call kmeans_mpi_cuda, gather and write results

#include <iostream>
#include <mpi.h>
#include "../utils/io.h"
#include "../utils/kmeans_common.h"
#include "kmeans_mpi_cuda.cuh"

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // TODO: rank 0 reads data, scatters to all ranks
    // TODO: all ranks call kmeans_mpi_cuda(...)
    // TODO: rank 0 gathers results and calls write_results(...)

    if (rank == 0) {
        std::cout << "MPI+CUDA K-Means: not yet implemented." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
