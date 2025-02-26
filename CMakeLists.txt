cmake_minimum_required(VERSION 3.10)
project(Solver CXX)

# Set mpic++ as the C++ compiler.
set(CMAKE_CXX_COMPILER mpic++)

# Append necessary compile flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -std=c++17")

# Define the source files.
set(SRCS
    src/main.cpp
    src/graph.cpp
    src/branch_and_bound.cpp
    src/globals.cpp
)

# Define separate variables for each directory.
set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)
set(SEQ_DIR ${CMAKE_BINARY_DIR}/output/sequential)
set(LOG_DIR ${CMAKE_BINARY_DIR}/output/log)
set(TXT_DIR ${CMAKE_BINARY_DIR}/output/outTxt)
set(PLOTS_DIR ${CMAKE_BINARY_DIR}/plots)

# Create both directories.
file(MAKE_DIRECTORY ${BIN_DIR})
file(MAKE_DIRECTORY ${SEQ_DIR})
file(MAKE_DIRECTORY ${LOG_DIR})
file(MAKE_DIRECTORY ${TXT_DIR})
file(MAKE_DIRECTORY ${PLOTS_DIR})

# Add the executable target and specify its output directory.
add_executable(solver ${SRCS})
set_target_properties(solver PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

# (Optional) Add a custom target for cleaning up generated files.
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaned up bin directory."
)

set(RUN_BENCHMARK_SCRIPT "${CMAKE_BINARY_DIR}/run_benchmarks.sh")

# Overwrite the benchmark script with the new content.
file(WRITE "${RUN_BENCHMARK_SCRIPT}" "#!/bin/bash\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# run_benchmarks.sh\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# This script submits SLURM jobs for each instance file in ../instances,\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# testing different numbers of MPI processes.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Each job executes the equivalent of:\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#   salloc -N <MPI_count> -n <MPI_count>\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#   srun -n <MPI_count> ./bin/solver ../instances/<file.col> 10000\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "TIME_LIMIT=10000.0\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "BIN_DIR=\"../build/bin\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "EXECUTABLE=\"\$BIN_DIR/solver\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "OUTPUT_DIR=\"../build/output\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "INSTANCE_DIR=\"../instances\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# MPI process counts to test\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "cpu_counts=(1 2 4 8 16 32 64)\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Create output directory if needed\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "mkdir -p \"\$OUTPUT_DIR\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Loop over all .col files in the instances folder\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "for file in \$(find \"\$INSTANCE_DIR\" -name \"*.col\"); do\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "  for np in \"\${cpu_counts[@]}\"; do\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    echo \"Submitting job for file \$file with \$np MPI processes...\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    sbatch <<EOF\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#!/bin/bash\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#SBATCH --job-name=solver_np\${np}\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#SBATCH --output=\${OUTPUT_DIR}/solver_np\${np}_%j.out\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#SBATCH --nodes=\${np}\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#SBATCH --ntasks=\${np}\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#SBATCH --time=03:00:00\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Run the solver using srun with np MPI processes\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "export OMP_NUM_THREADS=256\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "srun -n \${np} \${EXECUTABLE} \${file} \${TIME_LIMIT}\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "EOF\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "  done\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "done\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "echo \"Job submission complete.\"\n")

# Make the benchmark script executable.
file(CHMOD "${RUN_BENCHMARK_SCRIPT}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# ------------------------------------------------------------------------
# Add the Python launcher script that runs three other Python scripts.
# ------------------------------------------------------------------------
set(RUN_SCRIPTS "${CMAKE_BINARY_DIR}/run_scripts.py")

# Write the Python script content.
file(WRITE "${RUN_SCRIPTS}" "#!/usr/bin/env python3\n")
file(APPEND "${RUN_SCRIPTS}" "import subprocess\n\n")
file(APPEND "${RUN_SCRIPTS}" "def main():\n")
file(APPEND "${RUN_SCRIPTS}" "    # List of the three Python scripts to run.\n")
file(APPEND "${RUN_SCRIPTS}" "    scripts = [\"../scripts/convert.py\", \"../scripts/visualize_speedup.py\", \"../scripts/tab_gen.py\"]\n")
file(APPEND "${RUN_SCRIPTS}" "    \n")
file(APPEND "${RUN_SCRIPTS}" "    # Launch each script one after the other.\n")
file(APPEND "${RUN_SCRIPTS}" "    for script in scripts:\n")
file(APPEND "${RUN_SCRIPTS}" "        print(f\"Launching {script}...\")\n")
file(APPEND "${RUN_SCRIPTS}" "        result = subprocess.run([\"python3\", script], capture_output=True, text=True)\n")
file(APPEND "${RUN_SCRIPTS}" "        \n")
file(APPEND "${RUN_SCRIPTS}" "        print(f\"Output from {script}:\\n{result.stdout}\")\n")
file(APPEND "${RUN_SCRIPTS}" "        if result.stderr:\n")
file(APPEND "${RUN_SCRIPTS}" "            print(f\"Errors from {script}:\\n{result.stderr}\")\n")
file(APPEND "${RUN_SCRIPTS}" "\n")
file(APPEND "${RUN_SCRIPTS}" "if __name__ == \"__main__\":\n")
file(APPEND "${RUN_SCRIPTS}" "    main()\n")

# Make the Python script executable.
file(CHMOD "${RUN_SCRIPTS}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Optionally, add a custom target to run the Python launcher.
add_custom_target(run-scripts
    COMMAND python3 ${RUN_SCRIPTS}
    DEPENDS ${RUN_SCRIPTS}
    COMMENT "Running the Python scripts using run_scripts.py"
)
