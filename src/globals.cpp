#include "globals.hpp"

std::chrono::steady_clock::time_point startTime;
bool searchCompleted = true;
int mpi_rank = 0;
int mpi_size = 1;
std::ofstream logStream;
