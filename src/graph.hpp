#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <vector>
#include <unordered_set>
#include <utility>
#include <string>
#include <queue>
#include <sstream>
#include <fstream>

using namespace std;

const int INF = 1000000000;

// Structure to hold a complete coloring solution.
struct ColoringSolution {
    int numColors;
    vector<int> coloring;
    ColoringSolution();
};

// Graph class (sparse representation)
struct Graph {
    int n;         // current number of vertices (after merges)
    int orig_n;    // original number of vertices
    vector<unordered_set<int>> adj;  // sparse adjacency list
    vector<vector<int>> mapping;     // mapping[i] holds the original vertex IDs merged into vertex i

    Graph(int n_);
    Graph();

    // Merge two vertices (Zykov branch "same color")
    Graph mergeVertices(int i, int j) const;
    // Add an edge (Zykov branch "different color")
    Graph addEdge(int i, int j) const;
    // Bron–Kerbosch for maximum clique (used for a lower bound)
    pair<int, vector<int>> heuristicMaxClique() const;
    // DSATUR heuristic coloring (upper bound)
    pair<int, vector<int>> heuristicColoring() const;
};

// Reads a .col file (1-indexed vertices) and builds the graph.
Graph readGraphFromCOLFile(const string &filename);
// Finds connected components via BFS.
vector<vector<int>> findConnectedComponents(const Graph &g);
// Given a full graph and a set of vertex indices (a connected component),
// builds the corresponding subgraph.
Graph extractSubgraph(const Graph &fullG, const vector<int> &vertices);

#endif // GRAPH_HPP
