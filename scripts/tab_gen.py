import os
import pandas as pd

def parse_log_file(filename):
    """Parses a log file and extracts relevant configuration and performance information."""
    config_data = {}
    with open(filename, 'r') as f:
        for line in f:
            if ":" in line:
                key, value = line.split(":", 1)
                config_data[key.strip()] = value.strip()

    instance_name = config_data.get("problem_instance_file_name")
    mpi_processes = int(config_data.get("number_of_mpi_processes", 0))
    threads_per_process = int(config_data.get("number_of_threads_per_process", 0))
    wall_time = float(config_data.get("wall_time_sec", float('nan')))
    is_within_time_limit = config_data.get("is_within_time_limit") == "true"
    num_colors = int(config_data.get("number_of_colors", 0))

    return {
        "Instance": instance_name,
        "MPI Processes": mpi_processes,
        "Threads per Process": threads_per_process,
        "Wall Time (sec)": wall_time,
        "Within Time Limit": is_within_time_limit,
        "Colors": num_colors
    }

def create_configuration_table(log_files_dir):
    """Creates a table summarizing all configurations from log files with custom sorting."""
    configurations = []
    for filename in os.listdir(log_files_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(log_files_dir, filename)
            config_data = parse_log_file(filepath)
            configurations.append(config_data)

    df = pd.DataFrame(configurations)

    if not df.empty:
        # Define the desired MPI process order
        preferred_mpi_order = [1, 2, 4, 8, 16, 32, 64]

        # Group by instance name and then apply custom sorting
        sorted_groups = []
        grouped = df.groupby('Instance')
        for name, group in grouped:
            mpi_processes_in_group = sorted(group['MPI Processes'].unique())

            sorted_mpi_processes = []
            for proc in preferred_mpi_order:
                if proc in mpi_processes_in_group:
                    sorted_mpi_processes.append(proc)
                    mpi_processes_in_group.remove(proc)

            # Append any remaining MPI processes in ascending order
            sorted_mpi_processes.extend(sorted(mpi_processes_in_group))

            instance_sorted_group = pd.DataFrame()
            for proc in sorted_mpi_processes:
                instance_sorted_group = pd.concat([instance_sorted_group, group[group['MPI Processes'] == proc]])
            sorted_groups.append(instance_sorted_group)

        config_table_df_sorted = pd.concat(sorted_groups).reset_index(drop=True)
        return config_table_df_sorted
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    log_files_directory = "."  # Replace with the actual directory containing your log files
    config_table_df_sorted = create_configuration_table(log_files_directory)

    if not config_table_df_sorted.empty:
        print(config_table_df_sorted.to_string())  # Print the sorted table to console
        # Optionally save to CSV or other formats
        # config_table_df_sorted.to_csv("configuration_table_sorted.csv", index=False)
        # config_table_df_sorted.to_markdown("configuration_table_sorted.md", index=False) # Requires 'tabulate' package
        print("\nConfiguration table generated and sorted.")
    else:
        print("No log files found or no data extracted.")