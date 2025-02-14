# EuroHPC_Student_Challenge_2025_Team_6 x Graph Coloring Solver

## Overview

This project implements a parallel **Branch-and-Bound** algorithm for **Graph Coloring**, leveraging **MPI** and **OpenMP** to improve performance. The objective is to determine the **chromatic number** of a given graph while optimizing computational efficiency.

&nbsp;
## Features
- **Graph coloring using Branch-and-Bound**
- **Heuristics for clique and coloring estimation**
- **Parallel execution using MPI and OpenMP**
- **Benchmarking automation via `run_benchmarks.sh`**

&nbsp;
## Output Format
Each run generates an output file `<basename>.output` with details about the execution. The `.output` file contains:

- Unique problem instance name
- Number of vertices and edges
- Best solution found within the time limit
- Indicator if the solution is optimal
- Total computation time (including I/O)
- Resources used (number of cores and nodes)



&nbsp;
## Installation
### INSTALL.md
This file explains the installation and setup process.

&nbsp;
## **Team Members**
| Name | GitHub Account |
|------|---------------|
| **Alberto Taddei** | https://github.com/albtad01 |
| **Arianna Amadini** | https://github.com/Arianna0709 |
| **Mario Capodanno** | https://github.com/MarioCapodanno |
| **Valerio Grillo** | https://github.com/Valegrl |