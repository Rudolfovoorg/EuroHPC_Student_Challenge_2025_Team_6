#ifndef GLOBALS_HPP
#define GLOBALS_HPP

#include <chrono>
#include <fstream>

// Global timing and MPI variables.
extern std::chrono::steady_clock::time_point startTime;
extern bool searchCompleted;
extern int mpi_rank, mpi_size;
extern std::ofstream logStream;

#endif // GLOBALS_HPP
