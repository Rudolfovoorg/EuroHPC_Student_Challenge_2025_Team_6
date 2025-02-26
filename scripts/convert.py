import os

def rename_files_in_directory(input_dir: str, output_subdir: str = 'outTxt') -> None:
    """
    Renames all files ending with '.output' in the specified input directory by replacing 
    the extension with '.txt' and moves them to an output subdirectory.

    The function creates the output subdirectory if it does not exist and skips any file 
    that is the output subdirectory itself.

    Parameters:
        input_dir (str): The directory containing the files to rename.
        output_subdir (str): The name of the subdirectory where renamed files will be saved.
                             Defaults to 'outTxt'.

    Returns:
        None
    """
    # Define the full path to the output directory
    output_dir = os.path.join(input_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        # Skip the output directory itself
        if filename == output_subdir:
            continue

        # Process files with the .output extension
        if filename.endswith('.output'):
            old_filepath = os.path.join(input_dir, filename)
            new_filename = filename.rsplit('.output', 1)[0] + '.txt'
            new_filepath = os.path.join(output_dir, new_filename)
            
            # Rename and move the file
            os.rename(old_filepath, new_filepath)
            print(f'Renamed {filename} to {new_filename} from {input_dir} to {output_dir}')

if __name__ == "__main__":
    """
    Main execution block.
    
    Sets the input directory and invokes the file renaming function.
    """
    input_directory = '../build/output'
    rename_files_in_directory(input_directory)
