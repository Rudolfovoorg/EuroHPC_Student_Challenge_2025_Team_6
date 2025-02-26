import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(filename):
    """Parses a log file and extracts relevant information."""
    instance_name = None
    mpi_processes = None
    wall_time = None
    colors = None

    with open(filename, 'r') as f:
        for line in f:
            if "problem_instance_file_name:" in line:
                instance_name = line.split(":")[1].strip()
            elif "number_of_mpi_processes:" in line:
                mpi_processes = int(line.split(":")[1].strip())
            elif "wall_time_sec:" in line:
                wall_time = float(line.split(":")[1].strip())
            elif "number_of_colors:" in line:
                colors = int(line.split(":")[1].strip())

    return instance_name, mpi_processes, wall_time, colors

def analyze_performance_data(log_files_dir):
    """Analyzes performance data from log files and calculates speedup."""
    performance_data = {}
    min_colors = {}

    for filename in os.listdir(log_files_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(log_files_dir, filename)
            instance_name, mpi_processes, wall_time, num_colors = parse_log_file(filepath)

            if instance_name and mpi_processes and wall_time is not None:
                if instance_name not in performance_data:
                    performance_data[instance_name] = {}
                    min_colors[instance_name] = num_colors # Initialize min colors with first file read. Will be updated if a better solution is found

                performance_data[instance_name][mpi_processes] = wall_time
                min_colors[instance_name] = min(min_colors[instance_name], num_colors)


    speedup_data = {}
    for instance, mpi_times in performance_data.items():
        if 1 in mpi_times: # Ensure we have a baseline (1 MPI process) for speedup calculation
            speedup_data[instance] = {}
            base_time = mpi_times[1]
            for processes, time in mpi_times.items():
                speedup_data[instance][processes] = base_time / time
        else:
            print(f"Warning: No 1-MPI process data found for instance {instance}. Speedup calculation skipped.")

    return speedup_data, min_colors

def visualize_speedup(speedup_data, min_colors, output_dir="speedup_plots"):
    """Visualizes speedup for each instance and global speedup."""
    os.makedirs(output_dir, exist_ok=True)

    # Instance-specific speedup plots
    for instance, speedups in speedup_data.items():
        mpi_processes = sorted(speedups.keys())
        speedup_values = [speedups[p] for p in mpi_processes]

        plt.figure(figsize=(10, 6))
        plt.plot(mpi_processes, speedup_values, marker='o')
        plt.xlabel("Number of MPI Processes")
        plt.ylabel("Speedup (relative to 1 MPI process)")
        plt.title(f"Speedup for Instance: {instance} (Colors: {min_colors[instance]})")
        plt.xticks(mpi_processes)
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"speedup_{instance}.png"))
        plt.close()

    # Global speedup plot
    global_speedup = {}
    all_processes = set()
    for speedups in speedup_data.values():
        all_processes.update(speedups.keys())
    sorted_processes = sorted(list(all_processes))

    for p in sorted_processes:
        times_for_process = []
        for instance in speedup_data:
            if p in speedup_data[instance]:
                times_for_process.append(speedup_data[instance][p])
        if times_for_process: # avoid empty list if some process count is not present in any instances.
            global_speedup[p] = sum(times_for_process) / len(times_for_process)


    plt.figure(figsize=(10, 6))
    plt.plot(sorted_processes, [global_speedup[p] for p in sorted_processes], marker='o')
    plt.xlabel("Number of MPI Processes")
    plt.ylabel("Average Speedup")
    plt.title("Global Average Speedup Across Instances")
    plt.xticks(sorted_processes)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "global_speedup.png"))
    plt.close()

if __name__ == "__main__":
    log_files_directory = "."  # Replace with the actual directory containing your log files
    speedup_data, min_colors = analyze_performance_data(log_files_directory)
    output_dir = "speedup_plots" # Define output_dir here
    visualize_speedup(speedup_data, min_colors, output_dir)  
    print(f"Speedup plots saved to the '{output_dir}' directory")