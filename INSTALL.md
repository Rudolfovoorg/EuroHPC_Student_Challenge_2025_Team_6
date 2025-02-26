
# Installation Guide
This file explains the installation and setup process.

&nbsp;
## Prerequisites
Ensure the following dependencies are installed:
- **CMake** (≥ 3.10)
- **MPI** (e.g., OpenMPI or MPICH)
- **GCC/G++** (≥ C++17)
- **OpenMP**

On Vega supercomputer, load the following:

```sh
module load openmpi/4.1.2.1
```

&nbsp;
## Compilation
Navigate to the project root and run:

```sh
mkdir build
cd build
cmake ..
make
```

This generates the executable `solver` in `build/bin/`.

&nbsp;
## Running the Solver
To execute the solver manually:

```sh
mpirun -np <num_processes> ./bin/solver <input_file> <time_limit_sec>
```

Example:

```sh
mpirun -np 4 ./bin/solver ../instances/anna.col 500
```

&nbsp;
## Running Benchmarks

To automate the execution across multiple input files and process configurations, use:

```sh
./run_benchmarks.sh
```

This script:

1. Iterates through `instances/`
2. Runs `solver` with varying MPI process counts (1, 2, 4, 8, 16, 32, 64)
3. Saves results in `build/output/benchmark_results.csv`