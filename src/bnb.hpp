#ifndef BNB_HPP
#define BNB_HPP

#include <mpi.h>
#include <omp.h>
#include <vector>
#include <cstdint>
#include <chrono>
#include <string>
#include <fstream>

namespace bnb {

// Global variables.
extern int n;                          // number of vertices
extern std::vector<uint64_t> gmask;      // bit-level neighbor masks for each vertex
extern int best;                       // best (minimum) number of colors found
extern std::vector<int> bestColors;      // best complete coloring (if needed)
extern bool timeLimitReached;            // flag for time limit
extern std::chrono::steady_clock::time_point searchStartTime;
extern std::chrono::milliseconds timeLimit; // time limit
extern int globalLB;                   // global clique lower bound

// MPI-related globals.
extern int mpi_rank, mpi_size;

// Logging.
extern bool logEnabled;
extern std::ofstream logFile;

// Utility function.
inline int popcount(uint64_t x) {
    return __builtin_popcountll(x);
}

// Checks whether the time limit has been reached.
bool checkTime();

// DSATUR-based functions.
int selectVertexDSATUR(const std::vector<int>& colors);
int greedyDSATURBit();
int computeLowerBound(const std::vector<int>& colors, int currentMax);

// Maximum Clique Computation.
void cliqueSearch(uint64_t R, uint64_t P, uint64_t X, int &maxClique);
int computeMaxClique();

// Logging helper.
void logNode(const std::vector<int>& colors, int coloredCount, int currentMax);

// Branch-and-bound search with OpenMP tasks.
extern const int TASK_DEPTH_THRESHOLD;
void search(int coloredCount, int currentMax, std::vector<int>& colors, int depth);

// Subproblem structure and generation.
struct Subproblem {
    std::vector<int> colors;
    int coloredCount;
    int currentMax;
};
void generateSubproblems(int depth, std::vector<int>& colors, int coloredCount, int currentMax,
                         std::vector<Subproblem>& subs, int limit);

// Helper: Compute output file name from instance name.
std::string getOutputFileName(const std::string &instanceName);

} // namespace bnb

#endif // BNB_HPP
