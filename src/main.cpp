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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int mpiRank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    // Start the wall-clock timer.
    startTime = steady_clock::now();

    // Set the number of OpenMP threads per process.
    unsigned int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;
    omp_set_num_threads(numThreads);

    if (argc < 3) {
        if (mpiRank == 0) {
            std::cerr << "Usage: " << argv[0] << " <input_file> <time_limit_sec>\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string inputFile = argv[1];
    double timeLimit = atof(argv[2]);

    // Lambda to extract the base name from a file path.
    auto getBaseName = [&](const std::string &fileName) {
        size_t pos = fileName.find_last_of("/\\");
        std::string base = (pos == std::string::npos) ? fileName : fileName.substr(pos + 1);
        size_t dotPos = base.find_last_of('.');
        return (dotPos != std::string::npos) ? base.substr(0, dotPos) : base;
    };
    std::string baseName = getBaseName(inputFile);

    // Open a log file specific to this MPI process.
    // The log file name now includes the MPI rank, the instance's base name, and the total number of MPI processes.
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

    // Every MPI process reads the full graph.
    Graph fullGraph = readGraphFromCOLFile(inputFile);
    std::vector<std::vector<int>> components = findConnectedComponents(fullGraph);

    // Global final coloring.
    std::vector<int> globalColoring(fullGraph.orig_n, -1);
    int globalBestColors = INF;

    // If multiple components exist
    if (components.size() > 1) {
        int localBestColors = 0;
        std::vector<int> localColoring(fullGraph.orig_n, -1);

        for (size_t i = 0; i < components.size(); i++) {
            if (static_cast<int>(i % mpiSize) == mpiRank) {
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
        MPI_Reduce(&localBestColors, &globalBestColors, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(localColoring.data(), globalColoring.data(), fullGraph.orig_n, MPI_INT,
                   MPI_MAX, 0, MPI_COMM_WORLD);
    }
    else {
        // Single component, static decomposition
        Graph subG = extractSubgraph(fullGraph, components[0]);
        std::vector<Graph> tasks;
        ColoringSolution dummy;
        dummy.numColors = INF;

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

        struct { int value; int rank; } localPair, globalPair;
        localPair.value = localBestValue;
        localPair.rank  = mpiRank;
        MPI_Allreduce(&localPair, &globalPair, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        globalBestColors = globalBestValue;
        globalColoring.assign(fullGraph.orig_n, -1);

        if (mpiRank == globalPair.rank) {
            globalColoring = localBest.coloring;
        }
        MPI_Bcast(globalColoring.data(), fullGraph.orig_n, MPI_INT, globalPair.rank, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Close the log file before finishing.
    logStream.close();

    // Rank 0 writes final output to a .output file (including CPU topology summary).
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

        // Basic info
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

        // The final coloring
        for (int i = 0; i < fullGraph.orig_n; i++) {
            outFile << i << " " << globalColoring[i] << "\n";
        }

        outFile.close();
        std::cout << "Output written to " << outputFileName << std::endl;
    }

    MPI_Finalize();
    return 0;
}
