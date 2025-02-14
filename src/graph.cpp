#include "graph.hpp"
#include <iostream>
#include <sstream>
#include <algorithm>
#include <queue>

// --- ColoringSolution ---

ColoringSolution::ColoringSolution() : numColors(INF) {}

// --- Graph Constructors ---

Graph::Graph(int n_) : n(n_), orig_n(n_) {
    adj.resize(n);
    mapping.resize(n);
    for (int i = 0; i < n; i++) {
        mapping[i].push_back(i);
    }
}

Graph::Graph() : n(0), orig_n(0) {}

// --- Graph Member Functions ---

// Merge two vertices (Zykov branch "same color")
Graph Graph::mergeVertices(int i, int j) const {
    Graph newG(n - 1);
    newG.orig_n = orig_n;
    newG.adj.resize(newG.n);
    newG.mapping.resize(newG.n);

    // Build newIndices: all vertices except j.
    vector<int> newIndices;
    newIndices.reserve(n - 1);
    for (int k = 0; k < n; k++) {
        if (k == j) continue;
        newIndices.push_back(k);
    }

    // For the merged vertex (old index i), combine mappings.
    for (int a = 0; a < newG.n; a++) {
        int oldIndex = newIndices[a];
        if (oldIndex == i) {
            newG.mapping[a] = mapping[i];
            newG.mapping[a].insert(newG.mapping[a].end(), mapping[j].begin(), mapping[j].end());
        } else {
            newG.mapping[a] = mapping[oldIndex];
        }
    }

    // Rebuild the adjacency.
    for (int a = 0; a < newG.n; a++) {
        for (int b = a + 1; b < newG.n; b++) {
            int origA = newIndices[a];
            int origB = newIndices[b];
            bool connected = false;
            if (origA == i || origB == i) {
                if (origA == i)
                    connected = (adj[i].count(origB) || adj[j].count(origB));
                else
                    connected = (adj[origA].count(i) || adj[origA].count(j));
            } else {
                connected = (adj[origA].count(origB));
            }
            if (connected) {
                newG.adj[a].insert(b);
                newG.adj[b].insert(a);
            }
        }
    }
    return newG;
}

// Add an edge (Zykov branch "different color")
Graph Graph::addEdge(int i, int j) const {
    Graph newG = *this;
    if (i < n && j < n) {
        newG.adj[i].insert(j);
        newG.adj[j].insert(i);
    }
    return newG;
}

// --- Bron–Kerbosch Helper Function ---
static void bronKerbosch(const vector<unordered_set<int>> &adj,
                         vector<int> &R, vector<int> &P, vector<int> &X,
                         int &bestSize, vector<int> &bestClique) {
    if (P.empty() && X.empty()) {
        if ((int)R.size() > bestSize) {
            bestSize = R.size();
            bestClique = R;
        }
        return;
    }
    // Choose a pivot u from P ∪ X that maximizes |P ∩ N(u)|
    int pivot = -1;
    int maxCount = -1;
    vector<int> unionPX;
    unionPX.insert(unionPX.end(), P.begin(), P.end());
    unionPX.insert(unionPX.end(), X.begin(), X.end());
    for (int u : unionPX) {
        int count = 0;
        for (int w : P) {
            if (adj[u].count(w))
                count++;
        }
        if (count > maxCount) {
            maxCount = count;
            pivot = u;
        }
    }
    vector<int> pWithoutPivot;
    for (int v : P)
        if (!adj[pivot].count(v))
            pWithoutPivot.push_back(v);
    for (int v : pWithoutPivot) {
        R.push_back(v);
        vector<int> newP, newX;
        for (int w : P)
            if (adj[v].count(w))
                newP.push_back(w);
        for (int w : X)
            if (adj[v].count(w))
                newX.push_back(w);
        bronKerbosch(adj, R, newP, newX, bestSize, bestClique);
        R.pop_back();
        P.erase(remove(P.begin(), P.end(), v), P.end());
        X.push_back(v);
        if (P.empty())
            break;
    }
}

// DSATUR heuristic coloring (upper bound).
pair<int, vector<int>> Graph::heuristicColoring() const {
    int nLocal = n;
    vector<int> color(nLocal, -1);
    vector<int> saturation(nLocal, 0);
    vector<int> degree(nLocal, 0);
    for (int i = 0; i < nLocal; i++)
        degree[i] = adj[i].size();

    auto pickNextVertex = [&]() -> int {
        int bestV = -1, bestSat = -1, bestDeg = -1;
        for (int v = 0; v < nLocal; v++) {
            if (color[v] == -1) {
                if (saturation[v] > bestSat || (saturation[v] == bestSat && degree[v] > bestDeg)) {
                    bestV = v;
                    bestSat = saturation[v];
                    bestDeg = degree[v];
                }
            }
        }
        return bestV;
    };

    for (int step = 0; step < nLocal; step++) {
        int v = pickNextVertex();
        if (v == -1) break;
        vector<bool> used(nLocal, false);
        for (int w : adj[v])
            if (color[w] != -1)
                used[color[w]] = true;
        int c = 0;
        while (c < nLocal && used[c])
            c++;
        color[v] = c;
        for (int w : adj[v])
            if (color[w] == -1) {
                bool seesC = false;
                for (int x : adj[w])
                    if (color[x] == c) { seesC = true; break; }
                if (!seesC)
                    saturation[w]++;
            }
    }
    int usedColors = 0;
    for (int v = 0; v < nLocal; v++)
        usedColors = max(usedColors, color[v] + 1);
    return {usedColors, color};
}

// Bron–Kerbosch for maximum clique.
pair<int, vector<int>> Graph::heuristicMaxClique() const {
    vector<int> R, P, X;
    P.resize(adj.size());
    for (int i = 0; i < (int)adj.size(); i++)
        P[i] = i;
    int bestSize = 0;
    vector<int> bestClique;
    bronKerbosch(adj, R, P, X, bestSize, bestClique);
    return {bestSize, bestClique};
}

// --- Other Graph Functions ---

// Reads a .col file (1-indexed vertices) and builds the graph.
Graph readGraphFromCOLFile(const string &filename) {
    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file " << filename << endl;
        exit(1);
    }
    int n = 0, m = 0;
    string line;
    vector<pair<int,int>> edges;
    while (getline(infile, line)) {
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
            edges.emplace_back(u - 1, v - 1);
        }
    }
    Graph g(n);
    for (auto &edge : edges) {
        int u = edge.first, v = edge.second;
        if (u >= 0 && u < n && v >= 0 && v < n) {
            g.adj[u].insert(v);
            g.adj[v].insert(u);
        }
    }
    return g;
}

// Find connected components via BFS.
vector<vector<int>> findConnectedComponents(const Graph &g) {
    vector<vector<int>> components;
    vector<bool> visited(g.n, false);
    for (int start = 0; start < g.n; start++) {
        if (!visited[start]) {
            queue<int> Q;
            Q.push(start);
            visited[start] = true;
            vector<int> comp;
            comp.push_back(start);
            while (!Q.empty()) {
                int v = Q.front(); Q.pop();
                for (int w : g.adj[v]) {
                    if (!visited[w]) {
                        visited[w] = true;
                        Q.push(w);
                        comp.push_back(w);
                    }
                }
            }
            components.push_back(comp);
        }
    }
    return components;
}

// Given a full graph and a set of vertex indices (a connected component),
// build the corresponding subgraph.
Graph extractSubgraph(const Graph &fullG, const vector<int> &vertices) {
    Graph subG(vertices.size());
    subG.orig_n = fullG.orig_n;
    for (int i = 0; i < (int)vertices.size(); i++) {
        subG.mapping[i] = fullG.mapping[vertices[i]];
    }
    for (int i = 0; i < (int)vertices.size(); i++) {
        for (int j = i + 1; j < (int)vertices.size(); j++) {
            int oldi = vertices[i];
            int oldj = vertices[j];
            if (fullG.adj[oldi].count(oldj)) {
                subG.adj[i].insert(j);
                subG.adj[j].insert(i);
            }
        }
    }
    return subG;
}
