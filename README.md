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



Example of `.output` file

```sh
problem_instance_file_name: anna
cmd_line: <root>/EuroHPC_Student_Challenge_2025_Team_6/build/../build/bin/solver ../instances/anna.col 10000.0 
solver_version: v1.0.0
number_of_vertices: 138
number_of_edges: 493
time_limit_sec: 10000
number_of_mpi_processes: 1
number_of_threads_per_process: 256
wall_time_sec: 2.83112
is_within_time_limit: true
number_of_colors: 11
0 0
1 0
2 0
.. ..
```



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