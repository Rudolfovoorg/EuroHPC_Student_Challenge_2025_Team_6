import os
import pandas as pd

def parse_log_file(filename: str) -> dict:
    """
    Parses a log file and extracts configuration and performance information.
    
    The log file is expected to contain lines with key-value pairs separated by a colon.
    The following keys are extracted (if present):
      - problem_instance_file_name: Name of the problem instance.
      - number_of_mpi_processes: Number of MPI processes (converted to int).
      - number_of_threads_per_process: Number of threads per process (converted to int).
      - wall_time_sec: Wall time in seconds (converted to float).
      - is_within_time_limit: A flag indicating whether the run is within the time limit.
      - number_of_colors: Number of colors (converted to int).
    
    Parameters:
        filename (str): The path to the log file.
    
    Returns:
        dict: A dictionary containing the extracted data with keys:
            "Instance", "MPI Processes", "Threads per Process",
            "Wall Time (sec)", "Within Time Limit", and "Colors".
    """
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

def create_configuration_table(log_files_dir: str) -> pd.DataFrame:
    """
    Creates a sorted configuration table by parsing log files in a given directory.
    
    The function reads all files ending with '.txt' in the specified directory,
    extracts configuration data using `parse_log_file`, and compiles the results into
    a pandas DataFrame. The DataFrame is then sorted by instance name and MPI process counts.
    
    Custom Sorting:
      - The preferred order for MPI processes is defined as [1, 2, 4, 8, 16, 32, 64].
      - For each problem instance, the entries are first sorted according to this preferred order.
      - Any MPI process values not in the preferred list are appended in ascending order.
    
    Parameters:
        log_files_dir (str): The directory containing the log files.
    
    Returns:
        pd.DataFrame: A sorted DataFrame containing configuration data from the log files.
                      Returns an empty DataFrame if no log files are found.
    """
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

        sorted_groups = []
        grouped = df.groupby('Instance')
        for name, group in grouped:
            mpi_processes_in_group = sorted(group['MPI Processes'].unique())

            sorted_mpi_processes = []
            for proc in preferred_mpi_order:
                if proc in mpi_processes_in_group:
                    sorted_mpi_processes.append(proc)
                    mpi_processes_in_group.remove(proc)

            # Append any remaining MPI process values in ascending order
            sorted_mpi_processes.extend(sorted(mpi_processes_in_group))

            instance_sorted_group = pd.DataFrame()
            for proc in sorted_mpi_processes:
                instance_sorted_group = pd.concat(
                    [instance_sorted_group, group[group['MPI Processes'] == proc]]
                )
            sorted_groups.append(instance_sorted_group)

        config_table_df_sorted = pd.concat(sorted_groups).reset_index(drop=True)
        return config_table_df_sorted
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    """
    Main execution block.
    
    Reads log files from the specified directory, creates a sorted configuration table,
    and writes the table to a CSV file.
    
    Expected log files location: '../build/output/outTxt'
    Output CSV file: '../build/configuration_table_sorted.csv'
    """
    log_files_directory = "../build/output/outTxt"
    config_table_df_sorted = create_configuration_table(log_files_directory)

    if not config_table_df_sorted.empty:
        output_filepath = "../build/configuration_table_sorted.csv"
        config_table_df_sorted.to_csv(output_filepath, index=False)
        print(config_table_df_sorted.to_string())
        print(f"\nConfiguration table generated, sorted, and saved to {output_filepath}.")
    else:
        print("No log files found or no data extracted.")
