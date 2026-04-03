import math
import os
import numpy as np
import pandas as pd
from basic import (
    divide_by_threshold,
    fix_cop_outliers,
    remove_offset,
    rotate_coordinate_system,
    transform_pad_to_global,
    get_pad_center,
    quick_interpolate,
    VoltConverter
)
from readc3d import C3DReader


def export_to_trc(data_array: np.ndarray, metadata: dict, output_path: str, 
              rotate=True, rotate_axis='x', rotate_angle: float=90):
    """
    Convert numpy array to .trc file format for motion capture data

    Parameters
    ----------
    data_array : numpy.ndarray
        Input data array, shape (3, n_markers, n_frames). 
        Needs to be transposed after passing in to be in the format of (n_frames, n_markers, 3)
    metadata : dict
        Metadata containing marker names and frame rate
    output_path : str
        Path to save the output .trc file
    rotate : bool, optional
        Whether to rotate the coordinate system, by default True
    rotate_axis : str, optional
        Axis to rotate around ('x', 'y', or 'z'), by default 'x'
    rotate_angle : float, optional
        Angle to rotate in degrees, by default 90
    """
    if rotate:
        data_array_rotated = rotate_coordinate_system(data_array, rotate_axis, rotate_angle, 0)
        data_array_T = data_array_rotated.transpose(2, 1, 0)
    else:
        # Transpose to (n_frames, n_markers, 3)
        data_array_T = data_array.transpose(2, 1, 0)

    # Extract necessary metadata
    n_frames, n_markers, _ = data_array_T.shape
    marker_names = metadata['marker_labels']
    frame_rate = metadata['marker_freq']
    unit = metadata['marker_unit']
    origin_n_frames = metadata['original_end_frame']

    # Create header
    header_1 = '\t'.join(['PathFileType', '4', '(X/Y/Z)', output_path]) + '\n'
    header_2 = '\t'.join(['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', 'OrigDataRate', 'OrigDataStartFrame', 'OrigNumFrames']) + '\n'
    header_3 = '\t'.join([str(frame_rate), str(frame_rate), str(n_frames), str(n_markers), unit, str(frame_rate), '1', str(origin_n_frames)]) + '\n'

    # Create labels line
    labels_line_1 = '\t'.join(['Frame#', 'Time'] + [f"{marker}\t\t" for marker in marker_names]) + '\n'
    labels_line_2 = '\t'.join(['\t'] + ['\t'.join([f"{axis}{i+1}" for axis in ['X', 'Y', 'Z']]) for i in range(n_markers)]) + '\n'

    # Write to file
    with open(output_path, 'w') as f:
        f.write(header_1)
        f.write(header_2)
        f.write(header_3)
        f.write(labels_line_1)
        f.write(labels_line_2)
        f.write('\n')
        
        for i in range(n_frames):
            frame_idx = i + 1  # Frame index starts from 1 in .trc files
            cur_time = frame_idx / frame_rate
            frame_data = data_array_T[i,:,:].flatten()
            f.write('\t'.join([str(frame_idx), str(cur_time)] + frame_data.astype(str).tolist()) + '\n')
    f.close()


def export_to_sto(fp_data=None, freq_fp=None, pad_data=None, freq_pad=None, output_path='', 
                  rotate=True, rotate_axis='x', rotate_angle: float=90, 
                  moment_point=0, marker_data=None, metadata=None):
    """
    Converts force plate and boxing pad data into OpenSim .sto file.
    If both `fp_data`and `pad_data` are passed in, all data will be concatenated into a single output file.
    
    Parameters
    -----------
    fp_data: list | dict, optional
        Multiple force plates should be stored in the list by numeric order (i.e., plate 1, then plate 2, etc.)
    freq_fp: float | int, optional
        Sampling frequency of force plate data in Hz.
    pad_data: dict, optional
        Dict containing boxing pad data.
    freq_pad: float | int, optional
        Sampling frequency of boxing pad data in Hz.
    output_path: str, optional
        String path for the output file.
    rotate : bool, optional
        Whether to rotate the coordinate system, by default True
    rotate_axis : str, optional
        Axis to rotate around ('x', 'y', or 'z'), by default 'x'
    rotate_angle : float, optional
        Angle to rotate in degrees, by default 90
    moment_point: int, optional
        The point relative to which the moment (torque) values are expressed. \n
        0: The center of the equipment (force plate or boxing pad) \n
        1: The center of pressure
    marker_data : numpy.ndarray
        The marker data array for computing the center of the boxing pad
    metadata : dict, optional
        Metadata containing marker locations
    """
    columns = ['time']
    all_data_columns = []
    n_frames = 0

    if fp_data:
        # Ensure data is a list even if only one plate is provided
        if not isinstance(fp_data, list):
            fp_data = [fp_data]
            
        n_frames = fp_data[0]['force'].shape[1]
        time_array_fp = np.linspace(1 / freq_fp, n_frames / freq_fp, n_frames)
        all_data_columns.append(time_array_fp)

        # By convention, write force and COP columns for all bodies first, and write torque columns for all bodies at the end
        for i, plate_data in enumerate(fp_data):
            prefix = f"{i+1}_"
            # Standard OpenSim GRF labels
            labels = [
                f'{prefix}ground_force_vx', f'{prefix}ground_force_vy', f'{prefix}ground_force_vz',
                f'{prefix}ground_force_px', f'{prefix}ground_force_py', f'{prefix}ground_force_pz'
            ]
            columns.extend(labels)
            # Set moment point for ID computation
            if moment_point == 0 and metadata is not None:
                geo_center = np.mean(metadata['force_plate_corners'][:, :, i], axis=1, keepdims=True)
                moment_point_data = np.repeat(geo_center, repeats=n_frames, axis=1)
            else:
                moment_point_data = plate_data['center_of_pressure']
            # Stack the 3D components for this specific plate
            if rotate:
                plate_matrix = np.column_stack([
                    rotate_coordinate_system(plate_data['force'].T, rotate_axis, rotate_angle, 1),
                    rotate_coordinate_system(moment_point_data.T, rotate_axis, rotate_angle, 1)
                ])
            else:
                plate_matrix = np.column_stack([
                    plate_data['force'].T, 
                    moment_point_data.T
                ])
            all_data_columns.append(plate_matrix)
        for i, plate_data in enumerate(fp_data):
            prefix = f"{i+1}_"
            # Standard OpenSim GRF labels
            labels = [f'{prefix}ground_torque_x', f'{prefix}ground_torque_y', f'{prefix}ground_torque_z']
            columns.extend(labels)
            # Stack the 3D components for this specific plate
            if rotate:
                all_data_columns.append(rotate_coordinate_system(plate_data['moment'].T, rotate_axis, rotate_angle, 1))
            else:
                all_data_columns.append(plate_data['moment'].T)

    if pad_data:
        # Generate time array corresponding to boxing pad data if force plate data is missing
        if not fp_data:
            n_frames = pad_data['force'].shape[1]
            time_array_pad = np.linspace(1 / freq_pad, n_frames / freq_pad, n_frames)
            all_data_columns.append(time_array_pad)
        # data is array: (n_frames x 9) -> [Fx, Fy, Fz, Px, Py, Pz, Mx, My, Mz]
        columns.extend(['pad_force_vx', 'pad_force_vy', 'pad_force_vz', 
                        'pad_force_px', 'pad_force_py', 'pad_force_pz', 
                        'pad_torque_x', 'pad_torque_y', 'pad_torque_z'])
        if moment_point == 0 and marker_data is not None:
            moment_point_data = get_pad_center(marker_data, metadata)[4]
            moment_point_data = quick_interpolate(moment_point_data, pad_data['force'])
        else:
            moment_point_data = pad_data['center_of_pressure']
        if rotate:
            pad_columns = [
                rotate_coordinate_system(pad_data['force'].T, rotate_axis, rotate_angle, 1),
                rotate_coordinate_system(moment_point_data.T, rotate_axis, rotate_angle, 1),
                rotate_coordinate_system(pad_data['moment'].T, rotate_axis, rotate_angle, 1)
            ]
        else:
            pad_columns = [
                pad_data['force'].T,
                moment_point_data.T,
                pad_data['moment'].T
            ]
        if fp_data:
            # Resample the boxing pad data to match the frequency of force plate data
            # time_array_orig = np.linspace(1 / freq_pad, pad_data['force'].shape[1] / freq_pad, pad_data['force'].shape[1])
            for i, field_data in enumerate(pad_columns):
                pad_columns[i] = quick_interpolate(field_data, time_array_fp, time_axis=0)
                # pad_columns[i] = pchip_interpolate(time_array_orig, field_data, time_array_fp, axis=0)
        all_data_columns.append(np.column_stack(pad_columns))

    # Combine all columns into one matrix
    final_matrix = np.column_stack(all_data_columns)

    # Write the .sto file with the required header
    with open(output_path, 'w') as f:
        f.write(f"{output_path}\n")
        f.write("version=1\n")
        f.write(f"nRows={n_frames}\n")
        f.write(f"nColumns={len(columns)}\n")
        f.write("inDegrees=yes\n")
        f.write("endheader\n")
        
        df = pd.DataFrame(final_matrix, columns=columns)
        df.to_csv(f, sep='\t', index=False, lineterminator='\n')
    f.close()


def read_mocap_data(mocap_file_path, format='.c3d', c3dreader=None):
    """
    Read marker and force plate data from specified file path. Only .c3d file is supported.

    Parameters
    ----------
    mocap_file_path : str
        The path of the mocap file.
    format : str
        The file format. Only .c3d file is supported.
    c3dreader : C3DReader | object
        An initialized C3DReader object. If not passed in, it will be initialized when called.

    Returns
    -------
    mocap_data, mocap_metadata : dict, dict
        A dict storing marker data and force plate data, and a dict storing metadata.
    """
    if not os.path.exists(mocap_file_path):
        print(f"MoCap file {mocap_file_path} doesn't exist")
        mocap_data, mocap_metadata = None, None
    elif format == '.c3d':
        if not c3dreader:
            c3dreader = C3DReader()
        mocap_data, mocap_metadata = c3dreader.get_data_via_subprocess(mocap_file_path)
    else:
        mocap_data, mocap_metadata = None, None
    return mocap_data, mocap_metadata


def read_pad_data(pad_file_path, mocap_data, mocap_metadata, freq_pad, trial_idx=None, speed='', 
                  volt_converter=None, transform_to_global=True,
                  spectrum_subtract=True, silence=True, **kwargs):
    """
    Read boxing pad data from specified file path, and crop it to the same range as the mocap data.

    Parameters
    ----------
    pad_file_path : str
        The path of the boxing pad data file.
    mocap_data : dict
        A dict storing marker data and force plate data.
    mocap_metadata : dict
        A dict storing metadata of mocap data.
    freq_pad : int | float
        The sampling frequency of the pad data.
    trial_idx : int | float
        The index of the valid trials, not all the recorded trials.
    speed : str
        The speed of the trial.
    volt_converter : VoltConverter
        An initialized VoltConverter object. If not passed in, it will be initialized when called.
    transform_to_global : bool
        Whether or not to transform the data to be represented in global(lab) coordinate system. Default is `True`.
    
    Returns
    -------
    pad_data_dict : dict
        An array containing converted pad data.
    """
    if not os.path.exists(pad_file_path):
        print(f"Pad file {pad_file_path} doesn't exist")
        pad_data = None
    else:
        # Pad data need to be cropped to trial range
        pad_data = pd.read_csv(pad_file_path, sep='\s+', header=None).iloc[:, 1:]  # Exclude the first column (time)
        if not volt_converter:
            volt_converter = VoltConverter(freq_pad)
        if mocap_metadata:
            i_start_frame_pad = math.ceil((mocap_metadata['start_frame'] - 1)/ mocap_metadata['marker_freq'] * freq_pad)
            i_end_frame_pad = math.ceil(mocap_metadata['end_frame'] / mocap_metadata['marker_freq'] * freq_pad)
            pad_data = volt_converter.volt_to_mechanics(pad_data.iloc[i_start_frame_pad-1:i_end_frame_pad, :], spectrum_subtract, silence, **kwargs)
        elif mocap_data:
            print(f"MoCap metadata for trial {trial_idx} at speed {speed} is missing. The whole pad data is converted without cropping.")
            pad_data = volt_converter.volt_to_mechanics(pad_data, spectrum_subtract, silence, **kwargs)
        else:
            print(f"MoCap data for trial {trial_idx} at speed {speed} is missing. The whole pad data is converted without cropping.")
            pad_data = volt_converter.volt_to_mechanics(pad_data, spectrum_subtract, silence, **kwargs)
    if pad_data is not None:
        # Remove offset for calculating the COP
        pad_data_dict = {
            "force": remove_offset(pad_data[:, :3].T, freq_pad, -1, 'pad'),
            "moment": remove_offset(pad_data[:, 3:].T, freq_pad, -1, 'pad')    
        }
        # When Z-axis is perpendicular to the pad surface, the Z-coordinate of COP is approximated to zero
        pad_data_dict["center_of_pressure"] = np.row_stack([
            fix_cop_outliers(divide_by_threshold(-pad_data_dict['moment'][1, :], pad_data_dict['force'][2, :], 0.1),  # COP_x * Fz = -My => COP_x = -My / Fz
                             -0.1, 0.1, axis=1), 
            fix_cop_outliers(divide_by_threshold(pad_data_dict['moment'][0, :], pad_data_dict['force'][2, :], 0.1),   # COP_y * Fz = Mx => COP_y = Mx / Fz
                             -0.2, 0.2, axis=1),
            np.zeros(pad_data.shape[0])                                                                               # COP_z = 0
        ])
    else:
        pad_data_dict = None
    if transform_to_global:
        pad_data_dict = transform_pad_to_global(pad_data_dict, mocap_data['markers'], mocap_metadata)
    return pad_data_dict


def read_mot_sto(file_path):
    """
    Reads an OpenSim .mot or .sto file in to pandas DataFrame.

    Returns
    -------
    header (str): 
        The metadata block including 'endheader'.

    df (pd.DataFrame): 
        The numerical data.
    """
    header_lines = []
    reached_header_end = False
    with open(file_path, 'r') as f:
        lines = f.readlines()
        line_num = 0
        for i, line in enumerate(lines):
            if not reached_header_end:
                header_lines.append(line)
                if 'endheader' in line:
                    reached_header_end = True
                    line_num = i + 1
                    break
        f.close()
    
    header = "".join(header_lines)
    # Use io.StringIO to let pandas read the list of strings as a file
    df = pd.read_csv(file_path, sep='\t', skiprows=line_num)
    
    return header, df


def save_mot(file_path, header, df):
    """
    Saves a filtered DataFrame back into .mot format with the original header.
    """
    with open(file_path, 'w', newline='') as f:
        # 1. Write the original header
        f.write(header)
        
        # 2. Write the DataFrame as tab-separated values
        # index=False prevents pandas from adding an extra column
        df.to_csv(f, sep='\t', index=False, lineterminator='\n')
        f.close()