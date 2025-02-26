import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_log_file(filename: str) -> tuple:
    """
    Parses a log file and extracts relevant performance information.

    This function reads the log file line by line and extracts:
      - problem_instance_file_name: The instance name.
      - number_of_mpi_processes: The number of MPI processes.
      - wall_time_sec: The wall clock time (in seconds).
      - number_of_colors: The number of colors used.

    Parameters:
        filename (str): The full path to the log file.

    Returns:
        tuple: A tuple containing:
            - instance_name (str or None): The name of the problem instance.
            - mpi_processes (int or None): The number of MPI processes.
            - wall_time (float or None): The wall time in seconds.
            - colors (int or None): The number of colors.
    """
    instance_name = None
    mpi_processes = None
    wall_time = None
    colors = None

    with open(filename, 'r') as f:
        for line in f:
            if "problem_instance_file_name:" in line:
                instance_name = line.split(":", 1)[1].strip()
            elif "number_of_mpi_processes:" in line:
                mpi_processes = int(line.split(":", 1)[1].strip())
            elif "wall_time_sec:" in line:
                wall_time = float(line.split(":", 1)[1].strip())
            elif "number_of_colors:" in line:
                colors = int(line.split(":", 1)[1].strip())

    return instance_name, mpi_processes, wall_time, colors

def analyze_performance_data(log_files_dir: str) -> tuple:
    """
    Analyzes performance data from log files and calculates the speedup.

    For each log file in the specified directory (files ending with '.txt'),
    the function extracts the instance name, MPI process count, wall time, and
    number of colors. It then groups the data by instance and computes the speedup
    relative to the baseline (1 MPI process).

    If a log file for a given instance does not have data for 1 MPI process,
    the speedup calculation for that instance is skipped.

    Parameters:
        log_files_dir (str): The directory containing log files.

    Returns:
        tuple: A tuple containing:
            - speedup_data (dict): A dictionary mapping instance names to another
              dictionary mapping MPI process counts to speedup values.
            - min_colors (dict): A dictionary mapping instance names to the minimum
              number of colors found in the log files.
    """
    performance_data = {}
    min_colors = {}

    for filename in os.listdir(log_files_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(log_files_dir, filename)
            instance_name, mpi_processes, wall_time, num_colors = parse_log_file(filepath)

            if instance_name and mpi_processes and wall_time is not None:
                if instance_name not in performance_data:
                    performance_data[instance_name] = {}
                    min_colors[instance_name] = num_colors  # Initialize with the first encountered value

                performance_data[instance_name][mpi_processes] = wall_time
                # Update the minimum colors value if a lower value is found
                min_colors[instance_name] = min(min_colors[instance_name], num_colors)

    speedup_data = {}
    for instance, mpi_times in performance_data.items():
        if 1 in mpi_times:  # Ensure baseline exists for 1 MPI process
            speedup_data[instance] = {}
            base_time = mpi_times[1]
            for processes, time in mpi_times.items():
                speedup_data[instance][processes] = base_time / time
        else:
            print(f"Warning: No 1-MPI process data found for instance {instance}. Speedup calculation skipped.")

    return speedup_data, min_colors

def visualize_speedup(speedup_data: dict, min_colors: dict, output_dir: str) -> None:
    """
    Visualizes the speedup for each instance and the global average speedup.

    The function generates two sets of plots:
      1. Instance-specific speedup plots:
         - X-axis: Number of MPI Processes.
         - Y-axis: Speedup (relative to 1 MPI process).
         - Title includes the instance name and its minimum number of colors.
      2. A global average speedup plot:
         - X-axis: Number of MPI Processes.
         - Y-axis: Average speedup computed across all instances.

    All plots are saved to the specified output directory.

    Parameters:
        speedup_data (dict): Dictionary containing speedup data per instance.
        min_colors (dict): Dictionary containing the minimum number of colors for each instance.
        output_dir (str): The directory where the generated plots will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Generate instance-specific speedup plots
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

    # Generate a global average speedup plot
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
        if times_for_process:
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
    """
    Main execution block.

    This script analyzes performance data from log files located in a specified directory,
    calculates the speedup relative to the baseline (1 MPI process), and generates visualizations
    for both instance-specific and global average speedup. The resulting plots are saved to
    the designated output directory.

    Expected Input:
        Log files directory: "../build/output/outTxt"

    Output:
        Speedup plots are saved in the "../build/plots" directory.
    """
    # Input directory containing log files
    log_files_directory = "../build/output/outTxt"
    speedup_data, min_colors = analyze_performance_data(log_files_directory)
    
    # Output directory for the generated plots
    output_dir = "../build/plots"
    visualize_speedup(speedup_data, min_colors, output_dir)
    
    print(f"Speedup plots saved to the '{output_dir}' directory")
