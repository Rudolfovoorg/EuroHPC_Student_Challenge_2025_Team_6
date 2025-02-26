/**
 * @file main.cpp
 * @brief Main entry point for the parallel graph coloring solver.
 *
 * This file implements the main function for a parallel graph coloring solver that utilizes a branch-and-bound
 * algorithm for solving the graph coloring problem. The solver is designed to run on distributed systems using MPI
 * for inter-process communication and OpenMP for multi-threaded processing within each process.
 *
 * The program reads a graph from an input file (expected in .col format with 1-indexed vertices), partitions the
 * graph into connected components, and then applies a branch-and-bound search to determine a valid coloring using
 * the minimum number of colors. For graphs with multiple components, each component is processed independently across
 * MPI processes. For a single connected component, a static task decomposition strategy is used.
 *
 * @note The input graph file must be in .col format.
 *
 * @par Usage Example:
 * @code
 *   mpirun -np <num_processes> ./solver <input_file.col> <time_limit_sec>
 * @endcode
 */

 #include "globals.hpp"
 #include "graph.hpp"
 #include "branch_and_bound.hpp"
 
 #include <mpi.h>
 #include <omp.h>
 #include <iostream>
 #include <chrono>
 #include <fstream>
 #include <thread>
 #include <sstream>
 #include <string>
 #include <vector>
 #include <map>
 #include <set>
 #include <algorithm>
 #include <cstring>
 #include <cstdlib>
 #include <unistd.h>
 
 using std::chrono::duration_cast;
 using std::chrono::duration;
 using std::chrono::steady_clock;
 
 /**
  * @brief Main function that orchestrates the graph coloring process.
  *
  * This function initializes MPI and OpenMP, processes command-line arguments, reads the input graph,
  * partitions the graph into connected components, and applies the branch-and-bound algorithm to compute
  * a valid graph coloring solution. The work is distributed among MPI processes and OpenMP threads.
  *
  * Upon completion, the root process writes a detailed output file containing key information such as:
  * - The instance file name and command-line invocation
  * - Solver version and configuration details
  * - Problem size (number of vertices and edges)
  * - Timing and performance metrics
  * - The final coloring solution
  *
  * @param argc Number of command-line arguments (expected to be at least 3).
  * @param argv Array of command-line arguments:
  *             - argv[0]: Program name.
  *             - argv[1]: Path to the input graph file (.col format).
  *             - argv[2]: Time limit (in seconds) for the branch-and-bound search.
  *
  * @return int Returns 0 if execution is successful, or a non-zero value if an error occurs.
  *
  * @warning Ensure that the input file exists and that the time limit is a positive number.
  */
int main(int argc, char** argv) {
    // Initialize the MPI environment.
    MPI_Init(&argc, &argv);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Start the wall-clock timer.
    startTime = steady_clock::now();

    // Attempt to read the OMP_NUM_THREADS environment variable.
    const char* envThreads = std::getenv("OMP_NUM_THREADS");
    unsigned int numThreads = 0;
    if (envThreads) {
        numThreads = std::atoi(envThreads);
    }
    if (numThreads == 0) {
        numThreads = 1;  // Ensure at least one thread is used.
    }

    // Validate command-line arguments.
    if (argc < 3) {
        if (mpiRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_file> <time_limit_sec>\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string inputFile = argv[1];
    double timeLimit = atof(argv[2]);

    // Extract the base name (without directory or extension) from the input file path.
    auto getBaseName = [&](const std::string &fileName) -> std::string {
        size_t pos = fileName.find_last_of("/\\");
        std::string base = (pos == std::string::npos) ? fileName : fileName.substr(pos + 1);
        size_t dotPos = base.find_last_of('.');
        return (dotPos != std::string::npos) ? base.substr(0, dotPos) : base;
    };
    std::string baseName = getBaseName(inputFile);

    // Open a log file specific to this MPI process to record progress.
    {
        std::ostringstream logFileName;
        logFileName << "../build/output/log/branch_log_rank_" << mpiRank << ".txt";
        logStream.open(logFileName.str());
        if (!logStream) {
            std::cerr << "Error opening log file " << logFileName.str() << std::endl;
            MPI_Finalize();
            return 1;
        }
    }

    // Read the full graph from the input file.
    Graph fullGraph = readGraphFromCOLFile(inputFile);
    // Identify connected components within the graph.
    std::vector<std::vector<int>> components = findConnectedComponents(fullGraph);

    // Global variables to store the final coloring solution.
    std::vector<int> globalColoring(fullGraph.orig_n, -1);
    int globalBestColors = INF;

    // Process each connected component separately if more than one exists.
    if (components.size() > 1) {
        int localBestColors = 0;
        std::vector<int> localColoring(fullGraph.orig_n, -1);

        // Distribute connected components among MPI processes.
        for (size_t i = 0; i < components.size(); i++) {
            if (static_cast<int>(i % mpiSize) == mpiRank) {
                // Extract the subgraph corresponding to the current component.
                Graph subG = extractSubgraph(fullGraph, components[i]);
                ColoringSolution compBest;
                #pragma omp parallel
                {
                    #pragma omp single nowait
                    {
                        branchAndBound(subG, compBest, timeLimit, 0);
                    }
                }
                localBestColors = std::max(localBestColors, compBest.numColors);
                for (int v : components[i]) {
                    localColoring[v] = compBest.coloring[v];
                }
            }
        }
        // Reduce the results from all MPI processes.
        MPI_Reduce(&localBestColors, &globalBestColors, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(localColoring.data(), globalColoring.data(), fullGraph.orig_n, MPI_INT,
                MPI_MAX, 0, MPI_COMM_WORLD);
    }
    else {
        // For a single connected component, perform static task decomposition.
        Graph subG = extractSubgraph(fullGraph, components[0]);
        std::vector<Graph> tasks;
        ColoringSolution dummy;
        dummy.numColors = INF;

        // Decompose the search tree into smaller subproblems.
        decomposeBnb(subG, 0, 2, tasks, timeLimit, dummy);
        if (tasks.empty()) {
            tasks.push_back(subG);
        }

        ColoringSolution localBest;
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                for (size_t i = 0; i < tasks.size(); i++) {
                    if (static_cast<int>(i % mpiSize) == mpiRank) {
                        #pragma omp task firstprivate(i)
                        {
                            branchAndBound(tasks[i], localBest, timeLimit, 2);
                        }
                    }
                }
                #pragma omp taskwait
            }
        }

        int localBestValue = localBest.numColors;
        int globalBestValue;
        MPI_Allreduce(&localBestValue, &globalBestValue, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

        // Determine the MPI process with the best solution.
        struct { int value; int rank; } localPair, globalPair;
        localPair.value = localBestValue;
        localPair.rank  = mpiRank;
        MPI_Allreduce(&localPair, &globalPair, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        globalBestColors = globalBestValue;
        globalColoring.assign(fullGraph.orig_n, -1);

        // Broadcast the best coloring solution from the process that found it.
        if (mpiRank == globalPair.rank) {
            globalColoring = localBest.coloring;
        }
        MPI_Bcast(globalColoring.data(), fullGraph.orig_n, MPI_INT, globalPair.rank, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Close the log file.
    logStream.close();

    // The root process writes the final results to an output file.
    if (mpiRank == 0) {
        int edgeCount = 0;
        for (int i = 0; i < fullGraph.n; i++) {
            edgeCount += fullGraph.adj[i].size();
        }
        edgeCount /= 2;

        std::ostringstream cmdLine;
        for (int i = 0; i < argc; i++) {
            cmdLine << argv[i] << " ";
        }

        std::string outputDir = "../build/output/";
        std::string outputFileName = outputDir + baseName + "_" + std::to_string(mpiSize) + ".output";

        std::ofstream outFile(outputFileName);
        if (!outFile) {
            std::cerr << "Error opening output file " << outputFileName << std::endl;
            MPI_Finalize();
            return 1;
        }

        double wallTime = duration_cast<duration<double>>(steady_clock::now() - startTime).count();

        // Write detailed information about the instance and solution.
        outFile << "problem_instance_file_name: " << baseName << "\n";
        outFile << "cmd_line: " << cmdLine.str() << "\n";
        outFile << "solver_version: v1.0.0\n";
        outFile << "number_of_vertices: " << fullGraph.orig_n << "\n";
        outFile << "number_of_edges: " << edgeCount << "\n";
        outFile << "time_limit_sec: " << timeLimit << "\n";
        outFile << "number_of_mpi_processes: " << mpiSize << "\n";
        outFile << "number_of_threads_per_process: " << numThreads << "\n";
        outFile << "wall_time_sec: " << wallTime << "\n";
        outFile << "is_within_time_limit: " << (searchCompleted ? "true" : "false") << "\n";
        outFile << "number_of_colors: " << globalBestColors << "\n";

        // Output the final coloring assignment for each vertex.
        for (int i = 0; i < fullGraph.orig_n; i++) {
            outFile << i << " " << globalColoring[i] << "\n";
        }

        outFile.close();
        std::cout << "Output written to " << outputFileName << std::endl;
    }

    MPI_Finalize();
    return 0;
}
 