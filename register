
#!/usr/bin/python3

import os
import numpy as np
import nibabel as nib
import pandas as pd
import multiprocessing
from nilearn import input_data, datasets
from nipype.interfaces.fsl import FLIRT, FNIRT, ApplyWarp

# Load the Harvard-Oxford atlas
atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas_filename = atlas.maps
region_names = atlas.labels
custom_region_labels = atlas.labels

# Create a dictionary to map custom region labels to their atlas indices
custom_region_indices = {label: idx for idx, label in enumerate(region_names) if label in custom_region_labels}

# Define dataset directories
root_dir = 'dataset/non-user/'
output_dir = 'bold-time-series-csv-FINAL/non-user/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define MNI template path (FSL default)
MNI_TEMPLATE = "/usr/local/fsl/data/standard/MNI152_T1_2mm.nii.gz"

# Function to register anatomical image to MNI
def register_T1_to_MNI(t1w_path, output_prefix):
    """Registers T1w image to MNI space using FLIRT + FNIRT"""
    
    # Linear registration (FLIRT)
    flirt = FLIRT()
    flirt.inputs.in_file = t1w_path
    flirt.inputs.reference = MNI_TEMPLATE
    flirt.inputs.out_matrix_file = f"{output_prefix}_affine.mat"
    flirt.inputs.out_file = f"{output_prefix}_T1w_flirt.nii.gz"
    flirt.inputs.dof = 12
    flirt.run()

    # Nonlinear registration (FNIRT)
    fnirt = FNIRT()
    fnirt.inputs.in_file = t1w_path
    fnirt.inputs.affine_file = f"{output_prefix}_affine.mat"
    fnirt.inputs.ref_file = MNI_TEMPLATE
    fnirt.inputs.warped_file = f"{output_prefix}_T1w_fnirt.nii.gz"
    fnirt.inputs.field_file = f"{output_prefix}_warp.nii.gz"
    fnirt.run()

    return f"{output_prefix}_warp.nii.gz"

# Function to apply warp to functional image
def apply_warp_to_func(func_path, warp_file, output_path):
    """Applies nonlinear transformation to functional image"""

    applywarp = ApplyWarp()
    applywarp.inputs.in_file = func_path
    applywarp.inputs.ref_file = MNI_TEMPLATE
    applywarp.inputs.field_file = warp_file
    applywarp.inputs.out_file = output_path
    applywarp.run()

    return output_path

# Function to process each patient's data
def process_patient(patient_folder):
    anat_dir = os.path.join(root_dir, patient_folder, 'anat')
    func_dir = os.path.join(root_dir, patient_folder, 'func')

    # Process anatomical (T1w) data
    anat_files = [f for f in os.listdir(anat_dir) if f.endswith('_T1w.nii.gz')]
    if anat_files:
        t1w_path = os.path.join(anat_dir, anat_files[0])
        warp_file = register_T1_to_MNI(t1w_path, os.path.join(anat_dir, f"{patient_folder}_MNI"))
    else:
        print(f"No T1w found for {patient_folder}. Skipping registration.")
        return

    # Process functional (BOLD) data
    func_files = [f for f in os.listdir(func_dir) if f.endswith('_bold.nii.gz')]
    if func_files:
        func_path = os.path.join(func_dir, func_files[0])
        output_func_path = os.path.join(func_dir, f"{patient_folder}_func_MNI.nii.gz")
        registered_func_path = apply_warp_to_func(func_path, warp_file, output_func_path)

        # Extract BOLD signals using atlas
        masker = input_data.NiftiLabelsMasker(labels_img=atlas_filename, labels=custom_region_indices, standardize=True)
        masked_data = masker.fit_transform(registered_func_path)

        # Save extracted time series
        time_series_df = pd.DataFrame(masked_data, columns=custom_region_labels)
        output_csv_path = os.path.join(output_dir, f'{patient_folder}_bold_time_series.csv')
        time_series_df.to_csv(output_csv_path, index=False)

# Get list of patient folders
patient_folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

# Process in parallel
num_processes = multiprocessing.cpu_count()
with multiprocessing.Pool(processes=num_processes) as pool:
    pool.map(process_patient, patient_folders)
