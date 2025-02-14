#include "globals.hpp"
#include "graph.hpp"
#include "branch_and_bound.hpp"

#include <mpi.h>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <queue>
#include <algorithm>

using namespace std;
using namespace std::chrono;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Set the number of OpenMP threads per process.
    unsigned int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;
    omp_set_num_threads(numThreads);

    if (argc < 3) {
        if (mpi_rank == 0)
            cerr << "Usage: " << argv[0] << " <input_file> <time_limit_sec>\n";
        MPI_Finalize();
        return 1;
    }
    string inputFile = argv[1];
    double timeLimit = atof(argv[2]); // e.g., 10.0 seconds

    // Open a log file specific to this MPI process.
    {
        ostringstream logFileName;
        logFileName << "../build/output/log/branch_log_rank_" << mpi_rank << ".txt";
        logStream.open(logFileName.str());
        if (!logStream) {
            cerr << "Error opening log file " << logFileName.str() << endl;
            MPI_Finalize();
            return 1;
        }
    }

    // Start wall clock.
    startTime = steady_clock::now();

    // Every MPI process reads the full graph.
    Graph fullGraph = readGraphFromCOLFile(inputFile);
    vector<vector<int>> components = findConnectedComponents(fullGraph);

    // Global final coloring.
    vector<int> globalColoring(fullGraph.orig_n, -1);
    int globalBestColors = INF;

    // Two coarse–grain strategies:
    // (a) For multiple components, distribute them round–robin.
    // (b) For one giant component, use a static decomposition.
    if (components.size() > 1) {
        int localBestColors = 0;
        vector<int> localColoring(fullGraph.orig_n, -1);
        for (size_t i = 0; i < components.size(); i++) {
            if ((int)(i % mpi_size) == mpi_rank) {
                Graph subG = extractSubgraph(fullGraph, components[i]);
                ColoringSolution compBest;
                #pragma omp parallel
                {
                    #pragma omp single nowait
                    { branchAndBound(subG, compBest, timeLimit, 0); }
                }
                localBestColors = max(localBestColors, compBest.numColors);
                for (int v : components[i])
                    localColoring[v] = compBest.coloring[v];
            }
        }
        MPI_Reduce(&localBestColors, &globalBestColors, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(localColoring.data(), globalColoring.data(), fullGraph.orig_n, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    else {
        // One giant component: use static decomposition.
        Graph subG = extractSubgraph(fullGraph, components[0]);
        vector<Graph> tasks;
        ColoringSolution dummy;
        dummy.numColors = INF;
        decomposeBnb(subG, 0, 2, tasks, timeLimit, dummy);
        if (tasks.empty())
            tasks.push_back(subG); // fallback if decomposition pruned everything

        ColoringSolution localBest;
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                // Distribute tasks among MPI processes.
                for (size_t i = 0; i < tasks.size(); i++) {
                    if ((int)(i % mpi_size) == mpi_rank) {
                        #pragma omp task firstprivate(i)
                        { branchAndBound(tasks[i], localBest, timeLimit, 2); }
                    }
                }
                #pragma omp taskwait
            }
        }
        int localBestValue = localBest.numColors, globalBestValue;
        MPI_Allreduce(&localBestValue, &globalBestValue, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        struct { int value; int rank; } localPair, globalPair;
        localPair.value = localBestValue;
        localPair.rank  = mpi_rank;
        MPI_Allreduce(&localPair, &globalPair, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);
        globalBestColors = globalBestValue;
        globalColoring.assign(fullGraph.orig_n, -1);
        if (mpi_rank == globalPair.rank)
            globalColoring = localBest.coloring;
        MPI_Bcast(globalColoring.data(), fullGraph.orig_n, MPI_INT, globalPair.rank, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Close the log file before finishing.
    logStream.close();

    // Rank 0 writes final output.
    if (mpi_rank == 0) {
        double wallTime = duration_cast<duration<double>>(steady_clock::now() - startTime).count();
        int edgeCount = 0;
        for (int i = 0; i < fullGraph.n; i++)
            edgeCount += fullGraph.adj[i].size();
        edgeCount /= 2;

        ostringstream cmdLine;
        for (int i = 0; i < argc; i++)
            cmdLine << argv[i] << " ";

        auto getBaseName = [&](const string &fileName) {
            size_t pos = fileName.find_last_of("/\\");
            string base = (pos == string::npos) ? fileName : fileName.substr(pos + 1);
            size_t dotPos = base.find_last_of('.');
            return (dotPos != string::npos) ? base.substr(0, dotPos) : base;
        };
        string baseName = getBaseName(inputFile);

        // Set the output directory to the correct location.
        // Since run_benchmarks.sh is in project/src,
        // using "../build/output/" here resolves to project/build/output.
        string outputDir = "../build/output/";
        string outputFileName = outputDir + baseName + ".output";

        ofstream outFile(outputFileName);
        if (!outFile) {
            cerr << "Error opening output file " << outputFileName << endl;
            MPI_Finalize();
            return 1;
        }

        outFile << "problem_instance_file_name: " << baseName << "\n";
        outFile << "cmd_line: " << cmdLine.str() << "\n";
        outFile << "solver_version: v2.0.0_optimized\n";
        outFile << "number_of_vertices: " << fullGraph.orig_n << "\n";
        outFile << "number_of_edges: " << edgeCount << "\n";
        outFile << "time_limit_sec: " << timeLimit << "\n";
        outFile << "number_of_worker_processes: " << mpi_size << "\n";
        outFile << "number_of_cores_per_worker: " << numThreads << "\n";
        outFile << "wall_time_sec: " << wallTime << "\n";
        outFile << "is_within_time_limit: " << (searchCompleted ? "true" : "false") << "\n";
        outFile << "number_of_colors: " << globalBestColors << "\n";
        for (int i = 0; i < fullGraph.orig_n; i++)
            outFile << i << " " << globalColoring[i] << "\n";
        outFile.close();
        cout << "Output written to " << outputFileName << endl;
    }

    MPI_Finalize();
    return 0;

}
