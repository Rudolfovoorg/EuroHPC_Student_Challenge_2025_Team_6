// parallel_chromatic.cpp
#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <queue>
#include <chrono>
#include <limits>
#include <algorithm>
#include <cstdlib>
#include <string>
#include <set>
#include <thread>

using namespace std;
using namespace std::chrono;

const int INF = 1000000000;
steady_clock::time_point startTime;
bool searchCompleted = true;  // remains true if the complete search finished within time limit

// MPI global variables.
int mpi_rank, mpi_size;

// ----------------------------------------------------------------------
// Graph class: stores the graph using an adjacency matrix and a mapping
// from the current vertices to the original vertex IDs.
// ----------------------------------------------------------------------
struct Graph {
    int n;                // current number of vertices
    int orig_n;           // original number of vertices (remains constant)
    vector<vector<bool>> adj;      // adjacency matrix
    // mapping[i] is the list of original vertices represented by vertex i.
    vector<vector<int>> mapping;

    // Constructor for an initial graph with n vertices.
    Graph(int n_) : n(n_), orig_n(n_) {
        adj.assign(n, vector<bool>(n, false));
        mapping.resize(n);
        for (int i = 0; i < n; i++) {
            mapping[i].push_back(i);
        }
    }
    // Default constructor.
    Graph() : n(0), orig_n(0) {}
    
    // -------------------------------
    // Merge vertices i and j (assume i < j)
    // In the “same–color” branch the two vertices are merged.
    // The new graph has (n-1) vertices.
    // -------------------------------
    Graph mergeVertices(int i, int j) const {
        Graph newG(n - 1);
        newG.orig_n = orig_n;  // preserve original number of vertices
        newG.adj.assign(newG.n, vector<bool>(newG.n, false));
        newG.mapping.resize(newG.n);
        
        // Build a new ordering: skip vertex j.
        vector<int> newIndices;
        for (int k = 0; k < n; k++) {
            if (k == j) continue;
            newIndices.push_back(k);
        }
        // Build the new mapping.
        for (int a = 0; a < newG.n; a++) {
            int oldIndex = newIndices[a];
            if (oldIndex == i) {
                // For the merged vertex, union the mappings of i and j.
                newG.mapping[a] = mapping[i];
                newG.mapping[a].insert(newG.mapping[a].end(), mapping[j].begin(), mapping[j].end());
            } else {
                newG.mapping[a] = mapping[oldIndex];
            }
        }
        // Build the new adjacency matrix.
        for (int a = 0; a < newG.n; a++) {
            for (int b = a + 1; b < newG.n; b++) {
                int origA = newIndices[a];
                int origB = newIndices[b];
                bool connected = false;
                if (origA == i || origB == i) {
                    if (origA == i)
                        connected = (adj[i][origB] || adj[j][origB]);
                    else // origB == i
                        connected = (adj[origA][i] || adj[origA][j]);
                } else {
                    connected = adj[origA][origB];
                }
                newG.adj[a][b] = newG.adj[b][a] = connected;
            }
        }
        return newG;
    }

    // -------------------------------
    // Add an edge between vertices i and j.
    // -------------------------------
    Graph addEdge(int i, int j) const {
        Graph newG = *this;
        if(i < newG.n && j < newG.n)
            newG.adj[i][j] = newG.adj[j][i] = true;
        return newG;
    }

    // -------------------------------
    // Heuristic maximum clique (greedy):
    // Returns a pair: (size of clique, clique vertices).
    // This clique size is a lower bound on the chromatic number.
    // -------------------------------
    pair<int, vector<int>> heuristicMaxClique() const {
        vector<pair<int,int>> degVertex;
        for (int i = 0; i < n; i++) {
            int deg = 0;
            for (int j = 0; j < n; j++)
                if (adj[i][j])
                    deg++;
            degVertex.push_back({deg, i});
        }
        sort(degVertex.begin(), degVertex.end(), greater<pair<int,int>>());
        vector<int> clique;
        for (auto &p : degVertex) {
            int v = p.second;
            bool canAdd = true;
            for (int u : clique)
                if (!adj[v][u]) { canAdd = false; break; }
            if (canAdd)
                clique.push_back(v);
        }
        return { (int)clique.size(), clique };
    }

    // -------------------------------
    // Heuristic coloring (greedy):
    // Assigns to each vertex the smallest color not used by its already–colored neighbors.
    // Returns a pair: (number of colors used, coloring vector).
    // -------------------------------
    pair<int, vector<int>> heuristicColoring() const {
        vector<int> color(n, -1);
        for (int v = 0; v < n; v++) {
            vector<bool> used(n, false);
            for (int u = 0; u < n; u++) {
                if (adj[v][u] && color[u] != -1)
                    used[color[u]] = true;
            }
            int c;
            for (c = 0; c < n; c++)
                if (!used[c])
                    break;
            color[v] = c;
        }
        int numColors = *max_element(color.begin(), color.end()) + 1;
        return {numColors, color};
    }
};

// ----------------------------------------------------------------------
// Read a graph from a file.
// ----------------------------------------------------------------------
Graph readGraphFromCOLFile(const string &filename) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file " << filename << endl;
        exit(1);
    }
    
    int n = 0, m = 0;
    string line;
    vector<pair<int,int>> edges;
    
    while(getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == 'c') continue;
        if (line[0] == 'p') {
            istringstream iss(line);
            string tmp;
            iss >> tmp >> tmp >> n >> m;
        }
        if (line[0] == 'e') {
            istringstream iss(line);
            char e;
            int u, v;
            iss >> e >> u >> v;
            edges.push_back({u - 1, v - 1});
        }
    }
    
    Graph g(n);
    for (auto &edge : edges) {
        int u = edge.first, v = edge.second;
        if(u >= 0 && u < n && v >= 0 && v < n)
            g.adj[u][v] = g.adj[v][u] = true;
    }
    
    return g;
}

// ----------------------------------------------------------------------
// branchAndBound: recursively explore the search tree.
// Added parameter "depth" for MPI static distribution at the top level.
// ----------------------------------------------------------------------
void branchAndBound(const Graph &g, int &localBest, double timeLimit,
                    ofstream &logStream, vector<int> &bestColoring, int depth = 0) {
    // Check time limit.
    if (duration_cast<duration<double>>(steady_clock::now() - startTime).count() >= timeLimit) {
        searchCompleted = false;
        return;
    }

    // Compute heuristics.
    auto cliquePair = g.heuristicMaxClique();
    int lb = cliquePair.first;
    auto coloringPair = g.heuristicColoring();
    int ub = coloringPair.first;
    vector<int> currentColoring = coloringPair.second;

    // Log current node.
    logStream << "Node (" << g.n << " vertices): lb = " << lb << ", ub = " << ub << "\n";
    logStream << "  Clique: ";
    for (int v : cliquePair.second)
        logStream << v << " ";
    logStream << "\n  Coloring uses " << ub << " colors.\n";

    // Update best solution if found.
    if (ub < localBest) {
        localBest = ub;
        bestColoring.assign(g.orig_n, -1);
        for (int i = 0; i < g.n; i++) {
            for (int orig : g.mapping[i])
                bestColoring[orig] = currentColoring[i];
        }
        logStream << "  New local best: " << localBest << "\n";
    }

    // If bounds coincide then this branch is solved.
    if (lb == ub)
        return;
    // Prune if the lower bound is not better than the best known.
    if (lb >= localBest)
        return;

    // Select two non–adjacent vertices for branching.
    int v1 = -1, v2 = -1;
    bool found = false;
    for (int i = 0; i < g.n && !found; i++) {
        for (int j = i + 1; j < g.n; j++) {
            if (!g.adj[i][j]) {
                v1 = i;
                v2 = j;
                found = true;
                break;
            }
        }
    }
    if (!found)
        return; // no branching possible

    // Branch 1: merge v1 and v2 (force same color)
    Graph child1 = g.mergeVertices(v1, v2);
    // At depth==0, distribute branches among MPI processes.
    if (depth == 0) {
        static int branchIndex = 0;
        if ((branchIndex % mpi_size) == mpi_rank) {
            branchAndBound(child1, localBest, timeLimit, logStream, bestColoring, depth+1);
        }
        branchIndex++;
    } else {
        branchAndBound(child1, localBest, timeLimit, logStream, bestColoring, depth+1);
    }

    // Branch 2: add an edge between v1 and v2 (force different colors)
    Graph child2 = g.addEdge(v1, v2);
    if (depth == 0) {
        static int branchIndex = 1;  // continue counting (branch 1 was index 0)
        if ((branchIndex % mpi_size) == mpi_rank) {
            branchAndBound(child2, localBest, timeLimit, logStream, bestColoring, depth+1);
        }
        branchIndex++;
    } else {
        branchAndBound(child2, localBest, timeLimit, logStream, bestColoring, depth+1);
    }
}

// ----------------------------------------------------------------------
// Helper function: get base file name (without path or extension)
// ----------------------------------------------------------------------
string getBaseName(const string &fileName) {
    size_t pos = fileName.find_last_of("/\\");
    string base = (pos == string::npos) ? fileName : fileName.substr(pos + 1);
    size_t dotPos = base.find_last_of('.');
    if (dotPos != string::npos)
        return base.substr(0, dotPos);
    return base;
}

// ----------------------------------------------------------------------
// main()
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    // Initialize MPI.
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc < 3) {
        if(mpi_rank == 0)
            cerr << "Usage: " << argv[0] << " <input_file> <time_limit_sec>" << endl;
        MPI_Finalize();
        return 1;
    }
    string inputFile = argv[1];
    double timeLimit = atof(argv[2]);  // e.g., 10.0 seconds

    // Start wall clock measurement.
    startTime = steady_clock::now();

    // Each process opens its own log file.
    ostringstream oss;
    oss << "log_" << mpi_rank << ".txt";
    ofstream logStream(oss.str());
    if (!logStream) {
        cerr << "Error opening log file." << endl;
        MPI_Finalize();
        return 1;
    }

    // Read the input graph.
    Graph root = readGraphFromCOLFile(inputFile);

    int localBest = INF;          // best chromatic number found by this process
    vector<int> bestColoring;     // best coloring mapping for the original vertices (for this process)

    // Run branch-and-bound on the root graph.
    branchAndBound(root, localBest, timeLimit, logStream, bestColoring, 0);
    logStream.close();

    // Reduce bestSolution among processes.
    int globalBest;
    MPI_Reduce(&localBest, &globalBest, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    // Broadcast global best to all.
    MPI_Bcast(&globalBest, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Process 0 writes the output.
    if (mpi_rank == 0) {
        double wallTime = duration_cast<duration<double>>(steady_clock::now() - startTime).count();
        
        // For reporting, compute the number of edges.
        int edgeCount = 0;
        for (int i = 0; i < root.n; i++) {
            for (int j = i + 1; j < root.n; j++) {
                if (root.adj[i][j])
                    edgeCount++;
            }
        }
        
        ostringstream cmdLine;
        for (int i = 0; i < argc; i++) {
            cmdLine << argv[i] << " ";
        }
        
        unsigned int coresPerWorker = thread::hardware_concurrency();
        if (coresPerWorker == 0)
            coresPerWorker = 1;
        string baseName = getBaseName(inputFile);
        string outputFileName = baseName + ".output";
        
        ofstream outFile(outputFileName);
        if (!outFile) {
            cerr << "Error opening output file " << outputFileName << endl;
            MPI_Finalize();
            return 1;
        }
        outFile << "problem_instance_file_name: " << baseName << "\n";
        outFile << "cmd_line: " << cmdLine.str() << "\n";
        outFile << "solver_version: v1.0.1\n";
        outFile << "number_of_vertices: " << root.orig_n << "\n";
        outFile << "number_of_edges: " << edgeCount << "\n";
        outFile << "time_limit_sec: " << timeLimit << "\n";
        outFile << "number_of_worker_processes: " << mpi_size << "\n";
        outFile << "number_of_cores_per_worker: " << coresPerWorker << "\n";
        outFile << "wall_time_sec: " << wallTime << "\n";
        outFile << "is_within_time_limit: " << (searchCompleted ? "true" : "false") << "\n";
        outFile << "number_of_colors: " << globalBest << "\n";
        // Output the vertex-color mapping.
        for (int i = 0; i < root.orig_n; i++) {
            outFile << i << " " << bestColoring[i] << "\n";
        }
        outFile.close();
        cout << "Output written to " << outputFileName << endl;
    }

    MPI_Finalize();
    return 0;
}
