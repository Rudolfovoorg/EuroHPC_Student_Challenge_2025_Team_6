#include "bnb.hpp"

#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>
#include <string>
#include <cstdlib>

using namespace std;
using namespace std::chrono;
using namespace bnb;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Set the number of OpenMP threads per process.
    unsigned int numThreads = thread::hardware_concurrency();
    if (numThreads == 0) numThreads = 1;
    omp_set_num_threads(numThreads);

    auto overallStart = steady_clock::now();
    
    // Process command-line arguments.
    double inputTimeLimit = 10.0; // default 10 seconds
    for (int i = 2; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--log") {
            logEnabled = true;
        } else {
            try { inputTimeLimit = stod(arg); } catch (...) { }
        }
    }
    timeLimit = milliseconds(static_cast<int>(inputTimeLimit * 1000));
    
    if (logEnabled) {
        string logFileName = "output/log/bnb_log_rank" + to_string(mpi_rank) + ".txt";
        logFile.open(logFileName);
        if (!logFile) {
            cerr << "Error opening log file on rank " << mpi_rank << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    
    if (argc < 2) {
        if (mpi_rank == 0)
            cerr << "Usage: ./solver instance.col [timeLimitSeconds] [--log]" << endl;
        MPI_Finalize();
        return 1;
    }
    
    string instanceName = argv[1];
    vector<pair<int,int>> edges;
    int m; // number of edges
    if (mpi_rank == 0) {
        ifstream infile(instanceName);
        if (!infile) {
            cerr << "Error opening file " << instanceName << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        string line;
        n = 0;
        while(getline(infile, line)) {
            if (line.empty()) continue;
            if (line[0] == 'c') continue;
            if (line[0] == 'p') {
                istringstream iss(line);
                string tmp;
                iss >> tmp >> tmp >> n >> m;
            } else if (line[0] == 'e') {
                istringstream iss(line);
                char dummy;
                int u, v;
                iss >> dummy >> u >> v;
                edges.push_back({u - 1, v - 1});
            }
        }
        infile.close();
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Build graph.
    gmask.assign(n, 0ULL);
    if (mpi_rank == 0) {
        for (auto &edge: edges) {
            int u = edge.first, v = edge.second;
            gmask[u] |= (1ULL << v);
            gmask[v] |= (1ULL << u);
        }
    }
    MPI_Bcast(gmask.data(), n, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    
    // Compute initial upper bound.
    int initUB = greedyDSATURBit();
    best = initUB;
    
    // Compute maximum clique (global lower bound) on rank 0.
    if (mpi_rank == 0) {
        globalLB = computeMaxClique();
    }
    MPI_Bcast(&globalLB, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Generate subproblems to a fixed depth.
    vector<Subproblem> subs;
    int subDepthLimit = 6;
    if (mpi_rank == 0) {
        vector<int> initColors(n, 0);
        initColors.reserve(n);
        generateSubproblems(0, initColors, 0, 0, subs, subDepthLimit);
    }
    int numSubs;
    if (mpi_rank == 0)
        numSubs = subs.size();
    MPI_Bcast(&numSubs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Pack subproblems for MPI distribution.
    vector<int> subData(numSubs * (n + 2), 0);
    if (mpi_rank == 0) {
        for (int i = 0; i < numSubs; i++){
            subData[i * (n + 2)] = subs[i].coloredCount;
            subData[i * (n + 2) + 1] = subs[i].currentMax;
            for (int j = 0; j < n; j++){
                subData[i * (n + 2) + 2 + j] = subs[i].colors[j];
            }
        }
    }
    MPI_Bcast(subData.data(), numSubs * (n + 2), MPI_INT, 0, MPI_COMM_WORLD);
    
    if (mpi_rank != 0) {
        subs.resize(numSubs);
        for (int i = 0; i < numSubs; i++){
            subs[i].coloredCount = subData[i * (n + 2)];
            subs[i].currentMax = subData[i * (n + 2) + 1];
            subs[i].colors.resize(n);
            for (int j = 0; j < n; j++){
                subs[i].colors[j] = subData[i * (n + 2) + 2 + j];
            }
        }
    }
    
    searchStartTime = steady_clock::now();
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < numSubs; i++) {
        if (i % mpi_size == mpi_rank) {
            Subproblem sp = subs[i];
            vector<int> localColors = sp.colors;
            search(sp.coloredCount, sp.currentMax, localColors, subDepthLimit);
        }
    }
    
    int localOptimum = (!timeLimitReached) ? 1 : 0;
    int globalBest;
    MPI_Reduce(&best, &globalBest, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    int globalOptimum;
    MPI_Reduce(&localOptimum, &globalOptimum, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    
    if (logEnabled) {
        #pragma omp critical
        { logFile << "Final solution: " << best << " colors.\n"; }
        logFile.close();
    }
    
    auto overallEnd = steady_clock::now();
    double totalTime = duration_cast<duration<double>>(overallEnd - overallStart).count();

    int coresPerNode = omp_get_num_procs();
    int threadsPerNode = omp_get_max_threads();
    if (mpi_rank == 0) {
        string outFileName = getOutputFileName(instanceName);
        ofstream outFile(outFileName);
        if (!outFile) {
            cerr << "Error opening output file " << outFileName << endl;
        } else {

            outFile << "Instance,Size,Edges,BestSolution,Optimum,TotalTime(s),Threads_Per_Node,CPUs_Per_Node,Nodes\n";
            outFile << instanceName << ","
                    << n << ","
                    << m << ","
                    << globalBest << ","
                    << ((globalOptimum == 1) ? "Yes" : "No") << ","
                    << totalTime << ","
                    << threadsPerNode << "," 
                    << coresPerNode << ","
                    << mpi_size << "\n";
            outFile.close();
        }
    } 

    MPI_Finalize();
    return 0;

}
