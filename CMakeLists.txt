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
set(LOG_DIR ${CMAKE_BINARY_DIR}/output/log)

# Create both directories.
file(MAKE_DIRECTORY ${BIN_DIR})
file(MAKE_DIRECTORY ${LOG_DIR})

# Add the executable target and specify its output directory.
add_executable(solver ${SRCS})
set_target_properties(solver PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${BIN_DIR})

# (Optional) Add a custom target for cleaning up generated files.
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${BIN_DIR}
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaned up bin directory."
)

# Generation of the run_benchmarks.sh script in the build folder
set(RUN_BENCHMARK_SCRIPT "${CMAKE_BINARY_DIR}/run_benchmarks.sh")

file(WRITE "${RUN_BENCHMARK_SCRIPT}" "#!/bin/bash\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# run_benchmarks.sh\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# This script runs the solver for each benchmark file (with extension .col)\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# located in the specified folders. For each input file the solver is executed with\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# MPI process counts: 1, 2, 4, 8, 16, 32, and 64.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# The solver must be compiled and be available in the ../build/bin folder as \"solver\".\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# The solver accepts two arguments: <input_file> and <time_limit_sec>.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "#\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# The output file is expected to be written as <basename>.output in ../build/output.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Define the MPI process counts to test.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "cpu_counts=(1 2 4 8 16 32 64)\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Define the folders containing benchmark input files.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "folders=(\"../instances\")\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Set the time limit for each run (in seconds).\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "TIME_LIMIT=500.0\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Define the directories for the executable and output files.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "BIN_DIR=\"../build/bin\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "OUTPUT_DIR=\"../build/output\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Ensure the output directory exists.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "mkdir -p \"$OUTPUT_DIR\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Name of the MPI+OpenMP executable (change if necessary)\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "EXECUTABLE=\"$BIN_DIR/solver\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# CSV output file.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "OUTPUT_CSV=\"$OUTPUT_DIR/benchmark_results.csv\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Write CSV header.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "echo \"Folder,Input_File,CPU_Count,Problem_Instance,Vertices,Edges,Time_Limit,Worker_Processes,Cores_Per_Worker,Wall_Time,Within_Time_Limit,Colors\" > \"$OUTPUT_CSV\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "# Loop over the folders and input files.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "for folder in \"\${folders[@]}\"; do\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    # Check if folder exists.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    if [[ ! -d \"\$folder\" ]]; then\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        echo \"Folder \$folder does not exist. Skipping...\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        continue\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    fi\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    # For each .col file in the folder.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    for input_file in \"\$folder\"/*.col; do\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        # Skip if no .col files are found.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        if [[ ! -e \"\$input_file\" ]]; then\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            echo \"No .col files found in \$folder.\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            break\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        fi\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        base=\$(basename \"\$input_file\")\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        echo \"Processing benchmark: \$input_file\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        # For each requested MPI process count.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        for np in \"\${cpu_counts[@]}\"; do\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            echo \"  Running with \$np MPI processes...\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            \n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            # (Optional) Remove any previous output file so that we have a fresh run.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            output_file=\"$OUTPUT_DIR/\${base%.*}.output\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            if [[ -f \"\$output_file\" ]]; then\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "                rm \"\$output_file\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            fi\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            # Run the executable using mpirun.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            mpirun -np \"\$np\" \"\$EXECUTABLE\" \"\$input_file\" \"\$TIME_LIMIT\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            # Wait a moment for the file system to flush the output.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            sleep 1\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            # Check if the expected output file was created.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            if [[ ! -f \"\$output_file\" ]]; then\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "                echo \"    Warning: Output file \$output_file not found.\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "                continue\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            fi\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            # Parse key fields from the output file.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            problem_instance=\$(grep \"problem_instance_file_name:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            vertices=\$(grep \"number_of_vertices:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            edges=\$(grep \"number_of_edges:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            tlimit=\$(grep \"time_limit_sec:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            worker_processes=\$(grep \"number_of_worker_processes:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            cores=\$(grep \"number_of_cores_per_worker:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            wall_time=\$(grep \"wall_time_sec:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            within_time=\$(grep \"is_within_time_limit:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            colors=\$(grep \"number_of_colors:\" \"\$output_file\" | awk -F\": \" '{print \$2}')\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            # Append a new row to the CSV file.\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "            echo \"\$folder,\$base,\$np,\$problem_instance,\$vertices,\$edges,\$tlimit,\$worker_processes,\$cores,\$wall_time,\$within_time,\$colors\" >> \"$OUTPUT_CSV\"\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "        done\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "    done\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "done\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "\n")
file(APPEND "${RUN_BENCHMARK_SCRIPT}" "echo \"Benchmarking complete. Results saved in \$OUTPUT_CSV.\"\n")

# Make the script executable
file(CHMOD "${RUN_BENCHMARK_SCRIPT}" PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
