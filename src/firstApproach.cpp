// sequential_chromatic.cpp
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

// ----------------------------------------------------------------------
// Global constants and variables
// ----------------------------------------------------------------------
const int INF = 1000000000;
steady_clock::time_point startTime;
bool searchCompleted = true;  // remains true if the complete search finished within time limit

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
        if (line[0] == 'c') {
            // Comment line; skip it.
            continue;
        }
        if (line[0] == 'p') {
            // Expected format: "p edge n m"
            istringstream iss(line);
            string tmp;
            iss >> tmp;  // "p"
            iss >> tmp;  // "edge"
            iss >> n >> m;
        }
        if (line[0] == 'e') {
            // Expected format: "e u v" (typically 1-indexed)
            istringstream iss(line);
            char e;
            int u, v;
            iss >> e >> u >> v;
            // Convert from 1-indexed to 0-indexed.
            edges.push_back({u - 1, v - 1});
        }
    }
    
    // Create the graph.
    Graph g(n);
    for (auto &edge : edges) {
        int u = edge.first;
        int v = edge.second;
        if(u >= 0 && u < n && v >= 0 && v < n)
            g.adj[u][v] = g.adj[v][u] = true;
    }
    
    return g;
}

// ----------------------------------------------------------------------
// branchAndBound: recursively explore the search tree.
// It computes a heuristic clique (lower bound) and a greedy coloring (upper bound).
// When a new best (lower) coloring is found, the coloring for the original graph
// is computed using the mapping stored in the graph and saved in bestColoring.
// ----------------------------------------------------------------------
void branchAndBound(const Graph &g, int &localBest, double timeLimit,
                    ofstream &logStream, vector<int> &bestColoring) {
    // Check if time limit is reached.
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
        // Compute the coloring for the original vertices.
        bestColoring.assign(g.orig_n, -1);
        for (int i = 0; i < g.n; i++) {
            for (int orig : g.mapping[i]) {
                bestColoring[orig] = currentColoring[i];
            }
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
    branchAndBound(child1, localBest, timeLimit, logStream, bestColoring);

    // Branch 2: add an edge between v1 and v2 (force different colors)
    Graph child2 = g.addEdge(v1, v2);
    branchAndBound(child2, localBest, timeLimit, logStream, bestColoring);
}

// ----------------------------------------------------------------------
// Helper function: get the base file name (without path or extension).
// For example, "dir/problem_instance_xyz.txt" -> "problem_instance_xyz".
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
    // Expect command-line arguments: <input_file> <time_limit_sec>
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <input_file> <time_limit_sec>" << endl;
        return 1;
    }
    string inputFile = argv[1];
    double timeLimit = atof(argv[2]);  // e.g., 10.0 seconds

    // Start wall clock measurement.
    startTime = steady_clock::now();

    // Open a log file to record branch-and-bound progress.
    ofstream logStream("log.txt");
    if (!logStream) {
        cerr << "Error opening log file." << endl;
        return 1;
    }

    // Read the input graph.
    Graph root = readGraphFromCOLFile(inputFile);

    int localBest = INF;          // best chromatic number found
    vector<int> bestColoring;     // best coloring mapping for the original vertices

    // Run branch-and-bound on the root graph.
    branchAndBound(root, localBest, timeLimit, logStream, bestColoring);
    logStream.close();

    // Measure total wall time.
    double wallTime = duration_cast<duration<double>>(steady_clock::now() - startTime).count();

    // Compute the number of edges in the original graph.
    int edgeCount = 0;
    for (int i = 0; i < root.n; i++) {
        for (int j = i + 1; j < root.n; j++) {
            if (root.adj[i][j])
                edgeCount++;
        }
    }
    
    // Reconstruct the command line.
    ostringstream cmdLine;
    for (int i = 0; i < argc; i++) {
        cmdLine << argv[i] << " ";
    }
    
    // Determine the number of cores per worker.
    unsigned int coresPerWorker = std::thread::hardware_concurrency();
    if (coresPerWorker == 0)
        coresPerWorker = 1;

    // Get the base instance file name.
    string baseName = getBaseName(inputFile);
    string outputFileName = baseName + ".output";

    // Write the output in the required format.
    ofstream outFile(outputFileName);
    if (!outFile) {
        cerr << "Error opening output file " << outputFileName << endl;
        return 1;
    }
    
    outFile << "problem_instance_file_name: " << baseName << "\n";
    outFile << "cmd_line: " << cmdLine.str() << "\n";
    outFile << "solver_version: v1.0.1\n";
    outFile << "number_of_vertices: " << root.orig_n << "\n";
    outFile << "number_of_edges: " << edgeCount << "\n";
    outFile << "time_limit_sec: " << timeLimit << "\n";
    outFile << "number_of_worker_processes: 1\n";
    outFile << "number_of_cores_per_worker: " << coresPerWorker << "\n";
    outFile << "wall_time_sec: " << wallTime << "\n";
    outFile << "is_within_time_limit: " << (searchCompleted ? "true" : "false") << "\n";
    outFile << "number_of_colors: " << localBest << "\n";
    // Output the vertex-color mapping.
    for (int i = 0; i < root.orig_n; i++) {
        outFile << i << " " << bestColoring[i] << "\n";
    }
    outFile.close();

    cout << "Output written to " << outputFileName << endl;
    return 0;
}