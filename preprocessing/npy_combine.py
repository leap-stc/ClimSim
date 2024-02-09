import numpy as np
import os

def combine_npy_files(base_path, num_dirs, output_dir):
    combined_input = None
    combined_target = None
    
    for i in range(1, num_dirs + 1):  # Assuming directory names are 1-indexed
        input_path = os.path.join(base_path, str(i), 'train_input.npy')
        target_path = os.path.join(base_path, str(i), 'train_target.npy')
        
        # Load the current directory's data
        current_input = np.load(input_path)
        current_target = np.load(target_path)
        
        # Combine with previous data
        if combined_input is None:
            combined_input = current_input
            combined_target = current_target
        else:
            combined_input = np.concatenate((combined_input, current_input), axis=0)
            combined_target = np.concatenate((combined_target, current_target), axis=0)
    
    # Save the combined data
    combined_input_path = os.path.join(output_dir, 'train_input.npy')
    combined_target_path = os.path.join(output_dir, 'train_target.npy')
    np.save(combined_input_path, combined_input)
    np.save(combined_target_path, combined_target)

# Define the base path where the data directories are located
base_path = '/global/homes/z/zeyuanhu/scratch/hugging/E3SM-MMF_ne4/preprocessing/v3_adv_qlog_full'  # Change this to your actual base path
output_dir = base_path  # If you want to save in the same base directory, or specify a different directory
num_dirs = 8  # Number of directories containing train_input.npy and train_target.npy

combine_npy_files(base_path, num_dirs, output_dir)