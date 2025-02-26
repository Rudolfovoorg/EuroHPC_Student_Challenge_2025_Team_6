/**
 * @file branch_and_bound.cpp
 * @brief Implementation of branch-and-bound routines for graph coloring.
 */

 #include "branch_and_bound.hpp"
 #include "globals.hpp"
 
 #include <mpi.h>
 #include <omp.h>
 #include <chrono>
 #include <cstdlib>
 #include <iostream>
 #include <sstream>
 #include <algorithm>
 #include <thread>
 
 // Tuning parameters.
 static const int MIN_VERTICES_FOR_TASK = 30;  ///< Minimum vertices to spawn OpenMP tasks.
 static const int MAX_TASK_DEPTH       = 4;      ///< Maximum depth for fine–grain parallelism.
 static const int DECOMP_DEPTH         = 2;      ///< Depth to stop MPI-level decomposition.
 
 /**
  * @brief Selects a branching pair (two nonadjacent vertices with a high degree sum).
  *
  * Searches the graph for two vertices that are not adjacent and whose combined degree is maximal.
  *
  * @param g The graph.
  * @return A pair of vertex indices (v1, v2) chosen for branching.
  */
 std::pair<int,int> selectBranchingPair(const Graph &g) {
     int v1 = -1, v2 = -1, bestScore = -1;
     std::vector<int> degrees(g.n);
     for (int i = 0; i < g.n; i++)
         degrees[i] = g.adj[i].size();
     for (int i = 0; i < g.n; i++) {
         for (int j = i + 1; j < g.n; j++) {
             if (g.adj[i].find(j) == g.adj[i].end()) {
                 int score = degrees[i] + degrees[j];
                 if (score > bestScore) {
                     bestScore = score;
                     v1 = i;
                     v2 = j;
                 }
             }
         }
     }
     return {v1, v2};
 }
 
 /**
  * @brief Recursive branch-and-bound function for graph coloring.
  *
  * Explores the search space recursively using both merging and edge addition
  * strategies and updates the best solution.
  *
  * @param g The current graph.
  * @param bestSolution The best coloring solution found so far.
  * @param timeLimit Time limit for the search (in seconds).
  * @param depth Current recursion depth.
  */
 void branchAndBound(const Graph &g, ColoringSolution &bestSolution, double timeLimit, int depth) {
     if (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - startTime).count() >= timeLimit) {
         searchCompleted = false;
         return;
     }
     // Compute lower (clique) and upper (DSATUR) bounds.
     auto [lb, clique] = g.heuristicMaxClique();
     auto [ub, coloring] = g.heuristicColoring();
 
     // Log the current branch-and-bound node.
     {
         double currentTime = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - startTime).count();
         #pragma omp critical(log)
         {
             logStream << "Time: " << currentTime << " sec, Depth: " << depth
                       << ", Lower bound: " << lb << ", Clique: [";
             for (int v : clique)
                 logStream << v << " ";
             logStream << "], Upper bound: " << ub << ", Coloring: [";
             for (int c : coloring)
                 logStream << c << " ";
             logStream << "]" << std::endl;
         }
     }
 
     // Update best solution (critical section).
     #pragma omp critical
     {
         if (ub < bestSolution.numColors) {
             bestSolution.numColors = ub;
             bestSolution.coloring.assign(g.orig_n, -1);
             for (int i = 0; i < g.n; i++) {
                 for (int orig : g.mapping[i])
                     bestSolution.coloring[orig] = coloring[i];
             }
         }
     }
     if (lb == ub) return;
     if (lb >= bestSolution.numColors) return;
 
     // Select two nonadjacent vertices for branching.
     auto [v1, v2] = selectBranchingPair(g);
     if (v1 == -1) return;  // Graph is a clique.
 
     Graph childMerge = g.mergeVertices(v1, v2);
     Graph childEdge  = g.addEdge(v1, v2);
 
     bool doParallel = (g.n >= MIN_VERTICES_FOR_TASK) && (depth < MAX_TASK_DEPTH);
     if (doParallel) {
         #pragma omp task shared(bestSolution) firstprivate(childMerge, timeLimit, depth)
         { branchAndBound(childMerge, bestSolution, timeLimit, depth + 1); }
         #pragma omp task shared(bestSolution) firstprivate(childEdge, timeLimit, depth)
         { branchAndBound(childEdge, bestSolution, timeLimit, depth + 1); }
         #pragma omp taskwait
     } else {
         branchAndBound(childMerge, bestSolution, timeLimit, depth + 1);
         branchAndBound(childEdge, bestSolution, timeLimit, depth + 1);
     }
 }
 
 /**
  * @brief Decomposes the branch-and-bound search tree for MPI distribution.
  *
  * Explores the search tree up to a fixed depth and collects subproblems (tasks)
  * for distributed processing.
  *
  * @param g The current graph.
  * @param depth Current decomposition depth.
  * @param decompDepth Maximum depth for decomposition.
  * @param tasks Vector to store the generated subgraph tasks.
  * @param timeLimit Time limit for the search (in seconds).
  * @param dummySolution A dummy solution used for comparison.
  */
 void decomposeBnb(const Graph &g, int depth, int decompDepth,
                   std::vector<Graph> &tasks, double timeLimit,
                   const ColoringSolution &dummySolution) {
     if (std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - startTime).count() >= timeLimit)
         return;
     if (depth >= decompDepth) {
         tasks.push_back(g);
         return;
     }
     auto [lb, clique] = g.heuristicMaxClique();
     auto [ub, coloring] = g.heuristicColoring();
     if (lb == ub) return;
     if (lb >= dummySolution.numColors) return;
 
     auto [v1, v2] = selectBranchingPair(g);
     if (v1 == -1) return;
     Graph childMerge = g.mergeVertices(v1, v2);
     Graph childEdge  = g.addEdge(v1, v2);
 
     decomposeBnb(childMerge, depth + 1, decompDepth, tasks, timeLimit, dummySolution);
     decomposeBnb(childEdge, depth + 1, decompDepth, tasks, timeLimit, dummySolution);
 }
 