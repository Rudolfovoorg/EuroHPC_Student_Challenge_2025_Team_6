/**
 * @file branch_and_bound.hpp
 * @brief Declaration of branch-and-bound routines for graph coloring.
 */

 #ifndef BRANCH_AND_BOUND_HPP
 #define BRANCH_AND_BOUND_HPP
 
 #include "graph.hpp"
 #include <vector>
 
 /**
  * @brief Recursive branch-and-bound routine for graph coloring.
  *
  * Recursively explores the search space using the branch-and-bound strategy and
  * updates the best coloring solution.
  *
  * @param g The current graph.
  * @param bestSolution The best coloring solution found so far.
  * @param timeLimit Time limit for the search (in seconds).
  * @param depth Current recursion depth.
  */
 void branchAndBound(const Graph &g, ColoringSolution &bestSolution, double timeLimit, int depth = 0);
 
 /**
  * @brief Decomposes the branch-and-bound search tree for MPI distribution.
  *
  * Recursively explores the search tree up to a fixed depth and collects subproblems.
  *
  * @param g The current graph.
  * @param depth Current depth of decomposition.
  * @param decompDepth Maximum depth for decomposition.
  * @param tasks Vector to store generated subgraph tasks.
  * @param timeLimit Time limit for the search (in seconds).
  * @param dummySolution A dummy solution used for comparisons.
  */
 void decomposeBnb(const Graph &g, int depth, int decompDepth,
                   std::vector<Graph> &tasks, double timeLimit,
                   const ColoringSolution &dummySolution);
 
 /**
  * @brief Selects a branching pair of vertices (two nonadjacent vertices with high degree sum).
  * @param g The graph.
  * @return A pair of vertex indices to branch on.
  */
 std::pair<int,int> selectBranchingPair(const Graph &g);
 
 #endif // BRANCH_AND_BOUND_HPP
 