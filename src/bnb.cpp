#include "bnb.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <algorithm>
#include <chrono>
#include <thread>
#include <iomanip>
#include <cstdlib>

namespace bnb {

using namespace std;
using namespace std::chrono;

// --- Global variable definitions ---
int n;
vector<uint64_t> gmask;
int best;
vector<int> bestColors;
bool timeLimitReached = false;
steady_clock::time_point searchStartTime;
milliseconds timeLimit(10000);  // default 10 seconds
int globalLB = 0;

int mpi_rank, mpi_size;

bool logEnabled = false;
ofstream logFile;

const int TASK_DEPTH_THRESHOLD = 5;

// --- Utility Functions ---
bool checkTime() {
    return duration_cast<milliseconds>(steady_clock::now() - searchStartTime) >= timeLimit;
}

// --- DSATUR Functions ---
int selectVertexDSATUR(const vector<int>& colors) {
    int vertex = -1, maxSat = -1, maxDegree = -1;
    for (int i = 0; i < n; i++) {
        if (colors[i] == 0) {
            uint64_t used = 0;
            for (uint64_t mask = gmask[i]; mask; mask &= mask - 1) {
                int nb = __builtin_ctzll(mask);
                if (colors[nb])
                    used |= (1ULL << (colors[nb] - 1));
            }
            int sat = popcount(used);
            int degree = popcount(gmask[i]);
            if (sat > maxSat || (sat == maxSat && degree > maxDegree)) {
                maxSat = sat;
                maxDegree = degree;
                vertex = i;
            }
        }
    }
    return vertex;
}

int greedyDSATURBit() {
    vector<int> colors(n, 0);
    colors.reserve(n);
    for (int i = 0; i < n; i++) {
        int vertex = selectVertexDSATUR(colors);
        if (vertex == -1) break;
        uint64_t used = 0;
        for (uint64_t mask = gmask[vertex]; mask; mask &= mask - 1) {
            int nb = __builtin_ctzll(mask);
            if (colors[nb])
                used |= (1ULL << (colors[nb] - 1));
        }
        int c = 1;
        while (used & (1ULL << (c - 1))) c++;
        colors[vertex] = c;
    }
    int maxColor = 0;
    for (int c : colors)
        maxColor = max(maxColor, c);
    return maxColor;
}

int computeLowerBound(const vector<int>& colors, int currentMax) {
    int lb = currentMax;
    for (int i = 0; i < n; i++) {
        if (colors[i] == 0) {
            uint64_t used = 0;
            for (uint64_t mask = gmask[i]; mask; mask &= mask - 1) {
                int nb = __builtin_ctzll(mask);
                if (colors[nb])
                    used |= (1ULL << (colors[nb] - 1));
            }
            lb = max(lb, popcount(used) + 1);
        }
    }
    return lb;
}

// --- Maximum Clique Computation (Bron–Kerbosch with pivoting) ---
void cliqueSearch(uint64_t R, uint64_t P, uint64_t X, int &maxClique) {
    if (P == 0 && X == 0) {
        maxClique = max(maxClique, popcount(R));
        return;
    }
    uint64_t PuX = P | X;
    int u = __builtin_ctzll(PuX);
    uint64_t candidates = P & ~(gmask[u]);
    while (candidates) {
        int v = __builtin_ctzll(candidates);
        candidates &= candidates - 1;
        cliqueSearch(R | (1ULL << v), P & gmask[v], X & gmask[v], maxClique);
        P &= ~(1ULL << v);
        X |= (1ULL << v);
        if (popcount(R) + popcount(P) <= maxClique)
            return;
    }
}

int computeMaxClique() {
    int maxClique = 0;
    uint64_t all = (n == 64) ? ~0ULL : ((1ULL << n) - 1);
    cliqueSearch(0, all, 0, maxClique);
    return maxClique;
}

// --- Logging ---
void logNode(const vector<int>& colors, int coloredCount, int currentMax) {
    if (!logEnabled) return;
    int lb = computeLowerBound(colors, currentMax);
    ostringstream oss;
    oss << "Node: coloredCount=" << coloredCount
        << ", currentMax=" << currentMax
        << ", lowerBound=" << lb
        << ", best=" << best << "\n";
    #pragma omp critical
    { 
        logFile << oss.str(); 
    }
}

// --- Branch-and-Bound Search with OpenMP Tasks ---
static thread_local int recCounter = 0;
void search(int coloredCount, int currentMax, vector<int>& colors, int depth) {
    recCounter++;
    if ((recCounter % 100 == 0) && checkTime()) {
        timeLimitReached = true;
        return;
    }
    
    logNode(colors, coloredCount, currentMax);
    
    if (coloredCount == n) {
        #pragma omp critical
        {
            if (currentMax < best) {
                best = currentMax;
                bestColors = colors;
            }
        }
        return;
    }
    
    int lb = computeLowerBound(colors, currentMax);
    lb = max(lb, globalLB);
    if (lb >= best)
        return;
    
    int v = selectVertexDSATUR(colors);
    if (v == -1) return;
    
    uint64_t used = 0;
    for (uint64_t mask = gmask[v]; mask; mask &= mask - 1) {
        int nb = __builtin_ctzll(mask);
        if (colors[nb])
            used |= (1ULL << (colors[nb] - 1));
    }
    
    #pragma omp taskgroup
    {
        for (int c = 1; c <= currentMax; c++) {
            if (!(used & (1ULL << (c - 1)))) {
                int newMax = max(currentMax, c);
                if (depth < TASK_DEPTH_THRESHOLD) {
                    vector<int> newColors = colors; // copy current state
                    newColors[v] = c;
                    #pragma omp task firstprivate(newColors, coloredCount, newMax, depth)
                    {
                        search(coloredCount + 1, newMax, newColors, depth + 1);
                    }
                } else {
                    int saved = colors[v];
                    colors[v] = c;
                    search(coloredCount + 1, newMax, colors, depth + 1);
                    colors[v] = saved;
                }
            }
        }
        if (currentMax + 1 < best) {
            if (depth < TASK_DEPTH_THRESHOLD) {
                vector<int> newColors = colors;
                newColors[v] = currentMax + 1;
                #pragma omp task firstprivate(newColors, coloredCount, depth)
                {
                    search(coloredCount + 1, currentMax + 1, newColors, depth + 1);
                }
            } else {
                int saved = colors[v];
                colors[v] = currentMax + 1;
                search(coloredCount + 1, currentMax + 1, colors, depth + 1);
                colors[v] = saved;
            }
        }
    } // taskgroup barrier.
}

// --- Subproblem Generation ---
void generateSubproblems(int depth, vector<int>& colors, int coloredCount, int currentMax,
                         vector<Subproblem>& subs, int limit) {
    if (depth == limit) {
        subs.push_back({colors, coloredCount, currentMax});
        return;
    }
    int v = selectVertexDSATUR(colors);
    if (v == -1) {
        subs.push_back({colors, coloredCount, currentMax});
        return;
    }
    uint64_t used = 0;
    for (uint64_t mask = gmask[v]; mask; mask &= mask - 1) {
        int nb = __builtin_ctzll(mask);
        if (colors[nb])
            used |= (1ULL << (colors[nb] - 1));
    }
    for (int c = 1; c <= currentMax; c++) {
        if (!(used & (1ULL << (c - 1)))) {
            vector<int> newColors = colors;
            newColors[v] = c;
            generateSubproblems(depth + 1, newColors, coloredCount + 1, currentMax, subs, limit);
        }
    }
    vector<int> newColors = colors;
    newColors[v] = currentMax + 1;
    generateSubproblems(depth + 1, newColors, coloredCount + 1, currentMax + 1, subs, limit);
}

// --- Helper ---
string getOutputFileName(const string &instanceName) {
    // Prepend "../output/" so that the .output file goes to build/output.
    string outDir = "../build/output/";
    string base = instanceName;
    size_t pos = base.find_last_of('/');
    if (pos != string::npos)
        base = base.substr(pos+1);
    pos = base.find_last_of('.');
    if (pos != string::npos)
        base = base.substr(0, pos);
    return outDir + base + ".output";
}

} // namespace bnb
