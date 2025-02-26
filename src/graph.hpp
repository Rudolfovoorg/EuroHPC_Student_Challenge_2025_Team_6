/**
 * @file graph.hpp
 * @brief Declaration of the Graph and ColoringSolution structures and related functions.
 */

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
 
 /**
  * @brief A constant representing an infinity value for coloring.
  */
 const int INF = 1000000000;
 
 /**
  * @brief Structure to hold a complete graph coloring solution.
  */
 struct ColoringSolution {
     int numColors;         ///< Number of colors used in the solution.
     vector<int> coloring;  ///< Color assignment for each vertex.
 
     /**
      * @brief Default constructor. Initializes numColors to INF.
      */
     ColoringSolution();
 };
 
 /**
  * @brief A sparse graph representation.
  */
 struct Graph {
     int n;         ///< Current number of vertices (after merges).
     int orig_n;    ///< Original number of vertices.
     vector<unordered_set<int>> adj;  ///< Sparse adjacency list.
     vector<vector<int>> mapping;     ///< mapping[i] holds the original vertex IDs merged into vertex i.
 
     /**
      * @brief Constructs a graph with a given number of vertices.
      * @param n_ Number of vertices.
      */
     Graph(int n_);
 
     /**
      * @brief Default constructor.
      */
     Graph();
 
     /**
      * @brief Merges two vertices (Zykov branch "same color").
      *
      * Merges vertex i and vertex j into one vertex, combining their adjacency
      * and original vertex mappings.
      *
      * @param i Index of the first vertex.
      * @param j Index of the second vertex.
      * @return A new Graph with the specified vertices merged.
      */
     Graph mergeVertices(int i, int j) const;
 
     /**
      * @brief Adds an edge between two vertices (Zykov branch "different color").
      *
      * Creates a new graph that is identical to the current graph except that
      * an edge is added between vertices i and j.
      *
      * @param i Index of the first vertex.
      * @param j Index of the second vertex.
      * @return A new Graph with the added edge.
      */
     Graph addEdge(int i, int j) const;
 
     /**
      * @brief Heuristically computes the maximum clique using Bron–Kerbosch algorithm.
      * @return A pair containing the size of the clique and the vertices forming the clique.
      */
     pair<int, vector<int>> heuristicMaxClique() const;
 
     /**
      * @brief Colors the graph heuristically using the DSATUR algorithm.
      * @return A pair containing the number of colors used and the color assignment.
      */
     pair<int, vector<int>> heuristicColoring() const;
 };
 
 /**
  * @brief Reads a .col file (1-indexed vertices) and builds the corresponding graph.
  * @param filename Name of the input file.
  * @return A Graph constructed from the file.
  */
 Graph readGraphFromCOLFile(const string &filename);
 
 /**
  * @brief Finds connected components in a graph using BFS.
  * @param g The graph.
  * @return A vector of components, where each component is a vector of vertex indices.
  */
 vector<vector<int>> findConnectedComponents(const Graph &g);
 
 /**
  * @brief Extracts a subgraph corresponding to a set of vertices from the full graph.
  * @param fullG The full graph.
  * @param vertices Vector of vertex indices representing a connected component.
  * @return The subgraph corresponding to the specified vertices.
  */
 Graph extractSubgraph(const Graph &fullG, const vector<int> &vertices);
 
 #endif // GRAPH_HPP
 