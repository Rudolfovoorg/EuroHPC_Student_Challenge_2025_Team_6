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

// MPI global variables
int mpi_rank, mpi_size;

// ----------------------------------------------------------------------
// Bron–Kerbosch for Maximum Clique (exact):
// Returns the size of the maximum clique + the clique vertices.
// This helps get a tighter lower bound on the chromatic number.
// ----------------------------------------------------------------------
static void bronKerbosch(
    const vector<vector<bool>> &adj,
    vector<int> &R, vector<int> &P, vector<int> &X,
    int &bestSize, vector<int> &bestClique
) {
    if (P.empty() && X.empty()) {
        // Found a maximal clique
        if ((int)R.size() > bestSize) {
            bestSize = (int)R.size();
            bestClique = R;
        }
        return;
    }
    // Choose a pivot to reduce branching (pivot-based optimization).
    // For simplicity, pick first in P as pivot:
    int pivot = P[0];
    // Build a set of neighbors of pivot
    vector<int> neighbors;
    for (int v : P) {
        if (adj[pivot][v]) {
            neighbors.push_back(v);
        }
    }
    // The standard pivot-based Bron–Kerbosch: iterate over P \ N(pivot)
    // i.e. all vertices in P that are not neighbors of pivot
    // to reduce recursion.
    vector<int> p_without_neigh;
    for (int v : P) {
        if (!adj[pivot][v]) {
            p_without_neigh.push_back(v);
        }
    }
    for (int v : p_without_neigh) {
        // R ∪ {v}
        R.push_back(v);

        // Compute P ∩ N(v) and X ∩ N(v)
        vector<int> newP, newX;
        for (int w : P) {
            if (adj[v][w]) newP.push_back(w);
        }
        for (int w : X) {
            if (adj[v][w]) newX.push_back(w);
        }
        bronKerbosch(adj, R, newP, newX, bestSize, bestClique);

        // Remove v from R
        R.pop_back();
        // Move v from P to X
        P.erase(find(P.begin(), P.end(), v));
        X.push_back(v);

        if (P.empty()) break; // can stop earlier
    }
}

// A helper to run Bron–Kerbosch on the entire graph
pair<int, vector<int>> exactMaxClique(const vector<vector<bool>> &adj) {
    int n = (int)adj.size();
    vector<int> R, P, X;
    P.reserve(n);
    for (int i = 0; i < n; i++) {
        P.push_back(i);
    }
    int bestSize = 0;
    vector<int> bestClique;
    bronKerbosch(adj, R, P, X, bestSize, bestClique);
    return {bestSize, bestClique};
}

// ----------------------------------------------------------------------
// Graph class
// ----------------------------------------------------------------------
struct Graph {
    int n;                       // current number of vertices
    int orig_n;                  // original number of vertices
    vector<vector<bool>> adj;    // adjacency matrix
    // mapping[i] => original vertex IDs merged into this "super-vertex"
    vector<vector<int>> mapping;

    // Constructor for an initial graph with n vertices
    Graph(int n_) : n(n_), orig_n(n_) {
        adj.assign(n, vector<bool>(n, false));
        mapping.resize(n);
        for (int i = 0; i < n; i++) {
            mapping[i].push_back(i);
        }
    }
    // Default constructor
    Graph() : n(0), orig_n(0) {}

    // Merge two vertices (Zykov branch "same color")
    Graph mergeVertices(int i, int j) const {
        Graph newG(n - 1);
        newG.orig_n = orig_n;
        newG.adj.assign(newG.n, vector<bool>(newG.n, false));
        newG.mapping.resize(newG.n);

        // Build newIndices skipping j
        vector<int> newIndices;
        newIndices.reserve(n - 1);
        for (int k = 0; k < n; k++) {
            if (k == j) continue;
            newIndices.push_back(k);
        }

        // Merge i and j => union mappings
        for (int a = 0; a < newG.n; a++) {
            int oldIndex = newIndices[a];
            if (oldIndex == i) {
                newG.mapping[a] = mapping[i];
                newG.mapping[a].insert(newG.mapping[a].end(),
                                       mapping[j].begin(), mapping[j].end());
            } else {
                newG.mapping[a] = mapping[oldIndex];
            }
        }

        // Build new adjacency
        for (int a = 0; a < newG.n; a++) {
            for (int b = a + 1; b < newG.n; b++) {
                int origA = newIndices[a];
                int origB = newIndices[b];
                bool connected = false;
                if (origA == i || origB == i) {
                    // merged i, j
                    if (origA == i) {
                        connected = (adj[i][origB] || adj[j][origB]);
                    } else {
                        connected = (adj[origA][i] || adj[origA][j]);
                    }
                } else {
                    connected = adj[origA][origB];
                }
                newG.adj[a][b] = newG.adj[b][a] = connected;
            }
        }
        return newG;
    }

    // Add an edge (Zykov branch "different color")
    Graph addEdge(int i, int j) const {
        Graph newG = *this;
        if (i < n && j < n) {
            newG.adj[i][j] = newG.adj[j][i] = true;
        }
        return newG;
    }

    // -------------------------------------------
    // Improved maximum clique (exact via Bron–Kerbosch)
    // => a *tight* lower bound on chromatic number
    // -------------------------------------------
    pair<int, vector<int>> heuristicMaxClique() const {
        // We rename to "exact" but we keep same function name for minimal code changes
        return exactMaxClique(adj);
    }

    // -------------------------------------------
    // DSATUR heuristic coloring => better upper bound
    // -------------------------------------------
    pair<int, vector<int>> heuristicColoring() const {
        int nLocal = n;
        vector<int> color(nLocal, -1);
        vector<int> saturation(nLocal, 0); // # distinct neighbor colors
        vector<int> degree(nLocal, 0);

        // Precompute degrees
        for (int i = 0; i < nLocal; i++) {
            int degCount = 0;
            for (int j = 0; j < nLocal; j++) {
                if (adj[i][j]) degCount++;
            }
            degree[i] = degCount;
        }

        // pick vertex with highest saturation, tie-break on largest degree
        auto pickNextVertex = [&]() {
            int bestV = -1, bestSat = -1, bestDeg = -1;
            for (int v = 0; v < nLocal; v++) {
                if (color[v] == -1) {
                    if (saturation[v] > bestSat ||
                       (saturation[v] == bestSat && degree[v] > bestDeg)) {
                        bestV = v;
                        bestSat = saturation[v];
                        bestDeg = degree[v];
                    }
                }
            }
            return bestV;
        };

        // DSATUR main loop
        for (int step = 0; step < nLocal; step++) {
            int v = pickNextVertex();
            if (v == -1) break;

            // find smallest color not used by neighbors
            vector<bool> used(degree.size(), false);
            for (int w = 0; w < nLocal; w++) {
                if (adj[v][w] && color[w] != -1) {
                    used[color[w]] = true;
                }
            }
            int c = 0;
            while (c < nLocal && used[c]) c++;
            color[v] = c;

            // update saturation of neighbors
            for (int w = 0; w < nLocal; w++) {
                if (adj[v][w] && color[w] == -1) {
                    // check if w already sees color c
                    bool seesC = false;
                    for (int x = 0; x < nLocal; x++) {
                        if (adj[w][x] && color[x] == c) {
                            seesC = true;
                            break;
                        }
                    }
                    if (!seesC) {
                        saturation[w]++;
                    }
                }
            }
        }

        int usedColors = 0;
        for (int v = 0; v < nLocal; v++) {
            usedColors = max(usedColors, color[v] + 1);
        }
        return {usedColors, color};
    }
};

// ----------------------------------------------------------------------
// Read a .col file into a Graph
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
            // .col files typically 1-based, convert to 0-based
            edges.push_back({u - 1, v - 1});
        }
    }

    Graph g(n);
    for (auto &edge : edges) {
        int u = edge.first, v = edge.second;
        if (u >= 0 && u < n && v >= 0 && v < n) {
            g.adj[u][v] = g.adj[v][u] = true;
        }
    }
    return g;
}

// ----------------------------------------------------------------------
// branchAndBound: recursively explore the Zykov tree
// ----------------------------------------------------------------------
void branchAndBound(const Graph &g, int &localBest, double timeLimit,
                    ofstream &logStream, vector<int> &bestColoring, int depth = 0)
{
    // Check time limit
    if (duration_cast<duration<double>>(steady_clock::now() - startTime).count() >= timeLimit) {
        searchCompleted = false;
        return;
    }

    // 1) Exact (Bron–Kerbosch) max clique => lower bound
    auto cliquePair = g.heuristicMaxClique();
    int lb = cliquePair.first;

    // 2) DSATUR => upper bound
    auto coloringPair = g.heuristicColoring();
    int ub = coloringPair.first;
    vector<int> currentColoring = coloringPair.second;

    // Log current node
    logStream << "Node (" << g.n << " vertices): lb = " << lb << ", ub = " << ub << "\n";
    logStream << "  Clique: ";
    for (int v : cliquePair.second) logStream << v << " ";
    logStream << "\n  Coloring uses " << ub << " colors.\n";

    // Update best solution
    if (ub < localBest) {
        localBest = ub;
        bestColoring.assign(g.orig_n, -1);
        // Map child coloring back to original vertices
        for (int i = 0; i < g.n; i++) {
            for (int orig : g.mapping[i]) {
                bestColoring[orig] = currentColoring[i];
            }
        }
        logStream << "  New local best: " << localBest << "\n";
    }

    // If lb == ub, we found an exact solution for this subproblem
    if (lb == ub) return;
    // Prune if the lower bound >= best known
    if (lb >= localBest) return;

    // 3) Select two non–adjacent vertices for branching
    int v1 = -1, v2 = -1;
    int bestScore = -1;
    for (int i = 0; i < g.n; i++) {
        int deg_i = 0;
        for (int x = 0; x < g.n; x++) {
            if (g.adj[i][x]) deg_i++;
        }
        for (int j = i + 1; j < g.n; j++) {
            if (!g.adj[i][j]) {
                // not adjacent => must branch
                int deg_j = 0;
                for (int x = 0; x < g.n; x++) {
                    if (g.adj[j][x]) deg_j++;
                }
                int score = deg_i + deg_j;
                if (score > bestScore) {
                    bestScore = score;
                    v1 = i;
                    v2 = j;
                }
            }
        }
    }
    if (v1 == -1) {
        // No non-adjacent pair => graph is a clique => done
        return;
    }

    // Branch 1: merge (v1, v2) => same color
    {
        Graph child = g.mergeVertices(v1, v2);
        if (depth == 0) {
            static int branchIndex = 0;
            if ((branchIndex % mpi_size) == mpi_rank) {
                branchAndBound(child, localBest, timeLimit, logStream, bestColoring, depth+1);
            }
            branchIndex++;
        } else {
            branchAndBound(child, localBest, timeLimit, logStream, bestColoring, depth+1);
        }
    }

    // Branch 2: add edge (v1, v2) => different colors
    {
        Graph child = g.addEdge(v1, v2);
        if (depth == 0) {
            static int branchIndex = 1;
            if ((branchIndex % mpi_size) == mpi_rank) {
                branchAndBound(child, localBest, timeLimit, logStream, bestColoring, depth+1);
            }
            branchIndex++;
        } else {
            branchAndBound(child, localBest, timeLimit, logStream, bestColoring, depth+1);
        }
    }
}

// ----------------------------------------------------------------------
// Find connected components of a graph (using BFS or DFS).
// Returns a vector of components, each component is a list of vertices
// ----------------------------------------------------------------------
vector<vector<int>> findConnectedComponents(const Graph &g) {
    vector<vector<int>> components;
    vector<bool> visited(g.n, false);
    for (int start = 0; start < g.n; start++) {
        if (!visited[start]) {
            // BFS or DFS
            queue<int> Q;
            Q.push(start);
            visited[start] = true;
            vector<int> comp;
            comp.push_back(start);
            while (!Q.empty()) {
                int v = Q.front(); Q.pop();
                // check neighbors
                for (int w = 0; w < g.n; w++) {
                    if (g.adj[v][w] && !visited[w]) {
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

// ----------------------------------------------------------------------
// Build a subgraph from the given set of vertices (one connected component).
// This is used to color each connected component individually.
// ----------------------------------------------------------------------
Graph extractSubgraph(const Graph &fullG, const vector<int> &vertices) {
    // Create a subgraph with 'vertices.size()' vertices
    Graph subG(vertices.size());
    subG.orig_n = fullG.orig_n; // keep the same "original" dimension
    // Map index => oldID
    // We'll also track subG.mapping[] properly
    for (int i = 0; i < (int)vertices.size(); i++) {
        subG.mapping[i] = fullG.mapping[vertices[i]];
    }

    // Build adjacency
    for (int i = 0; i < (int)vertices.size(); i++) {
        for (int j = i+1; j < (int)vertices.size(); j++) {
            int oldi = vertices[i];
            int oldj = vertices[j];
            bool c = fullG.adj[oldi][oldj];
            subG.adj[i][j] = subG.adj[j][i] = c;
        }
    }
    return subG;
}

// ----------------------------------------------------------------------
// main()
// ----------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if (argc < 3) {
        if (mpi_rank == 0) {
            cerr << "Usage: " << argv[0] << " <input_file> <time_limit_sec>\n";
        }
        MPI_Finalize();
        return 1;
    }
    string inputFile = argv[1];
    double timeLimit = atof(argv[2]);  // e.g. 10.0 seconds

    // start wall clock
    startTime = steady_clock::now();

    // Each process opens its own log file
    ostringstream oss;
    oss << "log_" << mpi_rank << ".txt";
    ofstream logStream(oss.str());
    if (!logStream) {
        cerr << "Error opening log file." << endl;
        MPI_Finalize();
        return 1;
    }

    // Read the input graph
    Graph fullGraph = readGraphFromCOLFile(inputFile);

    // 1) Find connected components (if multiple, color each separately).
    // We'll gather the final color assignment in "globalColoring",
    // and track the best number_of_colors found.
    vector<int> globalColoring(fullGraph.orig_n, -1);
    int globalBestColors = 0;

    // find components in 'fullGraph'
    auto components = findConnectedComponents(fullGraph);

    // We will color each component separately. Because the graph is disconnected,
    // the total chromatic number is the max of the components' chromatic numbers.
    // Also, there's no conflict between them, so we can re-use color IDs if we want.
    // For simplicity, we won't offset color IDs per component; we use exactly
    // 'componentBest' colors in each. Then the final # colors = max across them.
    // Implementation: do a loop over components. For each component:
    // - Extract subGraph
    // - BnB
    // - Merge coloring into globalColoring
    // - Track maximum used color.

    // We'll do them in sequence. If you have many components, you could do them
    // in parallel or distribute among processes, but that is a deeper refactor.

    int colorOffset = 0; // We'll unify color sets by reusing [0..(component chroma-1)] for each component
    for (size_t idxComp = 0; idxComp < components.size(); idxComp++) {
        // Build subgraph
        Graph subG = extractSubgraph(fullGraph, components[idxComp]);

        // Solve BnB on subG
        int localBest = INF;
        vector<int> bestColoring; // size subG.orig_n, but effectively we only fill the subcomponent's mapping

        branchAndBound(subG, localBest, timeLimit, logStream, bestColoring, 0);

        // Reduce best solution among processes
        int compGlobalBest;
        MPI_Reduce(&localBest, &compGlobalBest, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
        // Broadcast so all ranks know it
        MPI_Bcast(&compGlobalBest, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Master rank merges color assignment into globalColoring
        if (mpi_rank == 0) {
            // 'compGlobalBest' is the # of colors for this subcomponent
            globalBestColors = max(globalBestColors, compGlobalBest);

            // bestColoring[] is the color for each original vertex, or -1 if not in subG
            // so we copy it into globalColoring for those vertices in subG
            // because subG.mapping[i] is the set of original vertices that i represents
            // but note subG.n is the # of "super-vertices" in subG
            // bestColoring should have length = subG.orig_n, but only the relevant
            // positions for subG's mapping are meaningful. We'll do a direct read:

            // For each "super-vertex i" in subG, bestColoring[...] = color index.
            // subG.mapping[i] are the original IDs.
            // So we just ensure that the color is placed in globalColoring.

            // But here's the tricky part: if you want to unify color sets across components
            // so the final coloring uses exactly 'max' colors overall, that's actually fine:
            // There's no adjacency between components, so we can reuse color IDs.
            // That means we do NOT shift color by colorOffset. We can keep them as is.
            // 'localBest' is the count of distinct colors actually used.

            for (int i = 0; i < subG.n; i++) {
                int c = bestColoring[i]; 
                // subG.mapping[i] => original vertices
                for (int origV : subG.mapping[i]) {
                    globalColoring[origV] = c; // no offset needed
                }
            }
        }

        // Next component
    }

    // Now we have "globalBestColors" as the final chromatic number (the maximum across all components).
    // All processes barrier to ensure consistency
    MPI_Barrier(MPI_COMM_WORLD);

    // Master rank writes final output
    if (mpi_rank == 0) {
        double wallTime = duration_cast<duration<double>>(steady_clock::now() - startTime).count();

        // compute # edges for reporting
        int edgeCount = 0;
        for (int i = 0; i < fullGraph.n; i++) {
            for (int j = i + 1; j < fullGraph.n; j++) {
                if (fullGraph.adj[i][j]) edgeCount++;
            }
        }

        // reconstruct command line
        ostringstream cmdLine;
        for (int i = 0; i < argc; i++) {
            cmdLine << argv[i] << " ";
        }

        unsigned int coresPerWorker = thread::hardware_concurrency();
        if (coresPerWorker == 0) coresPerWorker = 1;

        // Output file
        auto getBaseName = [&](const string &fileName){
            size_t pos = fileName.find_last_of("/\\");
            string base = (pos == string::npos) ? fileName : fileName.substr(pos + 1);
            size_t dotPos = base.find_last_of('.');
            if (dotPos != string::npos) return base.substr(0, dotPos);
            return base;
        };
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
        outFile << "solver_version: v2.0.0_optimized\n";
        outFile << "number_of_vertices: " << fullGraph.orig_n << "\n";
        outFile << "number_of_edges: " << edgeCount << "\n";
        outFile << "time_limit_sec: " << timeLimit << "\n";
        outFile << "number_of_worker_processes: " << mpi_size << "\n";
        outFile << "number_of_cores_per_worker: " << coresPerWorker << "\n";
        outFile << "wall_time_sec: " << wallTime << "\n";
        outFile << "is_within_time_limit: " << (searchCompleted ? "true" : "false") << "\n";
        outFile << "number_of_colors: " << globalBestColors << "\n";
        // Output the vertex-color mapping
        for (int i = 0; i < fullGraph.orig_n; i++) {
            outFile << i << " " << globalColoring[i] << "\n";
        }
        outFile.close();
        cout << "Output written to " << outputFileName << endl;
    }

    MPI_Finalize();
    return 0;
}
