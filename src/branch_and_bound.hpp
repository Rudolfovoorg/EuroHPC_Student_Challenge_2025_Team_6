#ifndef BRANCH_AND_BOUND_HPP
#define BRANCH_AND_BOUND_HPP

#include "graph.hpp"
#include <vector>

// Recursive branch-and-bound routine.
void branchAndBound(const Graph &g, ColoringSolution &bestSolution, double timeLimit, int depth = 0);
// Static decomposition: explore the BnB tree up to a fixed depth for MPI distribution.
void decomposeBnb(const Graph &g, int depth, int decompDepth,
                  std::vector<Graph> &tasks, double timeLimit,
                  const ColoringSolution &dummySolution);
// Helper: select a branching pair (two nonadjacent vertices with high degree sum)
std::pair<int,int> selectBranchingPair(const Graph &g);

#endif // BRANCH_AND_BOUND_HPP
