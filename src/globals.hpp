/**
 * @file globals.hpp
 * @brief Declaration of global variables for timing, MPI, and logging.
 */

 #ifndef GLOBALS_HPP
 #define GLOBALS_HPP
 
 #include <chrono>
 #include <fstream>
 
 /**
  * @brief Global start time of the program.
  */
 extern std::chrono::steady_clock::time_point startTime;
 
 /**
  * @brief Flag indicating whether the search completed within the time limit.
  */
 extern bool searchCompleted;
 
 /**
  * @brief MPI rank of the current process.
  */
 extern int mpi_rank;
 
 /**
  * @brief Total number of MPI processes.
  */
 extern int mpi_size;
 
 /**
  * @brief Global output log stream.
  */
 extern std::ofstream logStream;
 
 #endif // GLOBALS_HPP
 