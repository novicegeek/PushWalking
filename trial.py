import copy
import dataclasses
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import opensim as osim
import xml.etree.ElementTree as ET
from basic import (Filter, extract_push_range, remove_offset, 
                   get_nearest_index, get_vector_norm, integrate_interval, find_response_moment,
                   temp_chdir, suppress_cpp_stdout)
from file import export_to_trc, export_to_sto, read_mot_sto, save_mot


ANALYSIS_DIR = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis"


@dataclass
class Trial():
    """
    Instance of a single trial.
    
    :param mocap_data: The motion capture data.
    :param mocap_metadata: The metadata of the motion capture data.
    :param mocap_file_path: The file path the motion capture data.
    :param pad_data: The boxing pad data.
    :param pad_file_path: The file path of the boxing pad data.
    :param speed: The subject's walking speed in the trial. Can be 'slow', 'normal' or 'fast'.
    :param trial_idx: The index of the trial in the measurement session.
    :param freq_marker: The sampling frequency of the marker data.
    :param freq_fp: The sampling frequency of the GRF data.
    :param freq_pad: The sampling frequency of the boxing pad data.
    :param synced: Whether the boxing pad data has been automatically synchronized with the motion capture data.
    """
    mocap_data: dict
    mocap_metadata: dict
    mocap_file_path: str
    pad_data: dict
    pad_file_path: str
    # start_time: float
    # end_time: float
    total_trial_idx: int
    speed: str
    trial_idx: int
    intensity: str
    subject: object
    model: int = None
    freq_marker: int = dataclasses.field(init=False)
    freq_fp: int = dataclasses.field(init=False)
    freq_pad: int = dataclasses.field(init=False)
    synced: bool = True

    def __post_init__(self):
        self.freq_marker = self.mocap_metadata['marker_freq']
        self.freq_fp = self.mocap_metadata['fp_freq']
        self.freq_pad = self.subject.freq_pad
        self.ik_dir = os.path.join(self.subject.subject_ik_dir, os.path.splitext(os.path.basename(self.mocap_file_path))[0])
        self.id_dir = os.path.join(self.subject.subject_id_dir, os.path.splitext(os.path.basename(self.mocap_file_path))[0])
        if self.subject.model:
            self.model = self.subject.model[self.total_trial_idx]

    def analyze_trial(self, cutoff_marker=15, cutoff_fp=15, cutoff_pad=15, normalize_criterium=None, 
                      overwrite=True, solve_ik=True, solve_id=True, pause_analyze=[], inspect_moment=False):
        """
        Interface method to initiate the analysis pipeline.

        :param normalize_criterium: The criterium to normalize the step cycle. If `None`, no normalization will be done.
        :param overwrite: Whether to overwrite the existing processed output file. If `False`, the existing output file will be loaded directly.
        :param cutoff_marker: The critical frequency to apply Butterworth filter to marker data.
        :param cutoff_fp: The critical frequency to apply Butterworth filter to force plate data.
        :param cutoff_pad: The critical frequency to apply Butterworth filter to boxing pad data.
        :param solve_ik: Whether to solve Inverse Kinematics.
        :param solve_id: Whether to solve Inverse Dynamics.
        """
        if self.total_trial_idx in pause_analyze:
            pass
        data_offset = {'markers': self.mocap_data['markers'],
                       'fp1': remove_offset(self.mocap_data['fp1'], self.freq_fp, -1, 'force plate'),
                       'fp2': remove_offset(self.mocap_data['fp2'], self.freq_fp, -1, 'force plate'),
                       'pad': remove_offset(self.pad_data, self.freq_pad, -1, 'pad')}
        data_filtered = {'markers': Filter.butter_filt(data_offset['markers'], cutoff_marker, self.freq_marker),
                         'fp1': Filter.butter_filt(data_offset['fp1'], cutoff_fp, self.freq_fp),
                         'fp2': Filter.butter_filt(data_offset['fp2'], cutoff_pad, self.freq_pad),
                         'pad': Filter.butter_filt(data_offset['pad'], cutoff_pad, self.freq_pad)}
        cropped = [self._crop_by_events(data_filtered['markers'], self.freq_marker),
                   self._crop_by_events(data_filtered['fp1'], self.freq_fp),
                   self._crop_by_events(data_filtered['fp2'], self.freq_fp),
                   self._crop_by_events(data_filtered['pad'], self.freq_pad)]
        data_cropped = {source: cropped[i][0] for i, source in enumerate(('markers', 'fp1', 'fp2', 'pad'))}
        # Each event is a list in the order of time
        # The first element is event name, second element is the frame index (starting from 0) in the cropped data
        idx_event_cropped = {source: cropped[i][1] for i, source in enumerate(('markers', 'fp1', 'fp2', 'pad'))}
        self.data_normalized = self.normalize_step_cycle(data_cropped, normalize_criterium)

        # Convert and save the marker data to .trc format for OpenSim API
        trc_output_path = os.path.splitext(self.mocap_file_path)[0] + '.trc'
        if not os.path.exists(trc_output_path):
            export_to_trc(self.data_normalized['markers'], self.mocap_metadata, trc_output_path)
        elif overwrite:
            print(f"Output file {trc_output_path} already exists. Overwrite the file as overwrite is set to True.")
            export_to_trc(self.data_normalized['markers'], self.mocap_metadata, trc_output_path)
        else:
            print(f"Output file {trc_output_path} already exists. Load the existing file directly as overwrite is set to False.")

        # Convert and save the force plate and boxing pad data to .sto format for OpenSim API
        force_output_path = os.path.splitext(self.mocap_file_path)[0] + '.sto'
        if not os.path.exists(force_output_path):
            export_to_sto([self.data_normalized['fp1'], self.data_normalized['fp2']], self.freq_fp, 
                          self.data_normalized['pad'], self.freq_pad, force_output_path,
                          moment_point=0, marker_data=self.data_normalized['markers'], metadata=self.mocap_metadata)
        elif overwrite:
            print(f"Output file {force_output_path} already exists. Overwrite the file as overwrite is set to True.")
            export_to_sto([self.data_normalized['fp1'], self.data_normalized['fp2']], self.freq_fp, 
                          self.data_normalized['pad'], self.freq_pad, force_output_path,
                          moment_point=0, marker_data=self.data_normalized['markers'], metadata=self.mocap_metadata)
        else:
            print(f"Output file {force_output_path} already exists. Load the existing file directly as overwrite is set to False.")

        # Solve IK and ID using OpenSim API
        if solve_ik:
            self.ik_results = self.solve_ik(trc_output_path, overwrite, filter_output=True, cutoff_freq=cutoff_marker)
        elif os.path.exists(os.path.join(self.ik_dir, os.path.basename(self.ik_dir) + '_ik_output.mot')):
            self.ik_results = os.path.join(self.ik_dir, os.path.basename(self.ik_dir) + '_ik_output.mot')
        if solve_id:
            self.id_results = self.solve_id(force_output_path, overwrite)
        elif os.path.exists(os.path.join(self.id_dir, os.path.basename(self.id_dir) + '_id_generalized_forces.sto')):
            self.id_results = os.path.join(self.id_dir, os.path.basename(self.id_dir) + '_id_generalized_forces.sto')
        
        return self.summarize_trial(idx_event_cropped, inspect_moment)
    
    def normalize_step_cycle(self, data, criterium: str = 'marker') -> dict | np.ndarray:
        """
        Normalize the time series data to percentage of gait cycle.
        
        :param data: The input data.
        :param criterium: Which data to use as the criterium for detection of ground contact of the non-dominant foot preceding the target step. Can be `'marker'`, `'grf'`, or `None`. \n
            If `'marker'`, the heel marker's vertical position is used to identify ground contact. \n
            If `'grf'`, the vertical GRF is used to identify ground contact. \n
            If `None`, no normalization is performed and the original data is returned.
        :return: Dict or numpy.ndarray of the normalized data.
        """
        if criterium is None:
            return data
        elif criterium == 'marker' or criterium == 'grf':
            data_normalized = {}
            for instrument, instrument_data in data.items():
                if isinstance(instrument_data, list):  # Force plate data
                    instrument_data_normalized = []
                    for field_data in instrument_data:
                        instrument_data_normalized.append(self._normalize_step_cycle(field_data, criterium))
                else:
                    instrument_data_normalized = self._normalize_step_cycle(instrument_data, criterium)
                data_normalized[instrument] = instrument_data_normalized
            return data_normalized
        else:
            raise ValueError("Invalid criterium for time normalization. Please choose from 'marker', 'grf', or None.")
    
    def _normalize_step_cycle(self, data, criterium: str = 'marker'):
        """
        (Unfinished method) Normalize a single data variable to percentage of step cycle.
        """
        data_normalized = copy.deepcopy(data)
        non_dominant_contact_frame = detect_ground_contact(data, criterium)[-1]
        dominant_contact_frame = detect_ground_contact(data, 'grf')[0]
        n_frames_step = dominant_contact_frame - non_dominant_contact_frame
        data_normalized[:, 0] = (data[:, 0] - non_dominant_contact_frame) / n_frames_step * 100

        return data_normalized

    def get_velocity(self, data) -> np.ndarray:
        """
        Calculate the movement velocity using the COM trajectory as approximated by the pelvis markers.

        :param data: The marker data. The default shape is 4xNxT (components XYZ + 1)x(number of markers)x(number of frames).
        """
        use_markers = ('R_ASIS', 'L_ASIS', 'R_PSIS', 'L_PSIS')
        labels = self.mocap_metadata['marker_labels']
        use_markers_idx = [labels.index(marker) for marker in use_markers]  # Get the column indices of the markers to use in the data
        COM_trajectory = np.mean(data[:3, use_markers_idx, :], axis=1)  # Average the XYZ components of the selected markers， result shape: XYZ x number of frame

        time_step = 1 / self.freq_marker
        velocity = np.gradient(COM_trajectory, time_step, axis=-1)
        velocity = np.vstack([velocity, np.linalg.norm(velocity, axis=0)])  # Calculate the overall velocity
        if self.mocap_metadata['marker_unit'] == 'mm':
            velocity = velocity / 1000
        return velocity, COM_trajectory

    def solve_ik(self, marker_file_path, overwrite, filter_output=True, cutoff_freq=10):
        """
        Solve inverse kinematics of the trial by calling OpenSim API.
        
        :param marker_file_path: The file path of the .trc file containing the marker data.
        :param overwrite: Whether to overwrite the existing IK output file.
        """

        print(f"Start solving IK for trial {self.trial_idx} at {self.speed} speed")
        if not os.path.exists(self.ik_dir):
            os.makedirs(self.ik_dir)
        output_motion_file_path = os.path.join(self.ik_dir, os.path.basename(self.ik_dir) + '_ik_output.mot')
        if os.path.exists(output_motion_file_path) and not overwrite:
            print(f"IK output file {output_motion_file_path} already exists. Load the existing file directly as overwrite is set to False.")
            return output_motion_file_path
        elif os.path.exists(output_motion_file_path):
            print(f"IK output file {output_motion_file_path} already exists. Overwrite the file as overwrite is set to True.")
        # osim.Logger.setLevelString('Warn')
        log_file_path = os.path.join(self.ik_dir, 'opensim.log')
        
        # Temporarily change the working directory to the trial's IK directory to be compatible with the relative path 
        # in the setup file, and suppress the output of the OpenSim API to accelerate the process.
        with temp_chdir(self.subject.subject_ik_dir), suppress_cpp_stdout(log_file_path):
            osim.Logger.removeFileSink()
            osim.Logger.addFileSink(log_file_path)
            osim.Logger.setLevelString('Info')

            # if not hasattr(self.subject, 'ik_solver') or not self.subject.ik_solver:
            if not self.model:
                ik_solver = osim.InverseKinematicsTool(self.subject.subject_ik_setup)
            else:
                trial_ik_setup = os.path.join(self.subject.subject_ik_dir, self.subject.subject_id + f"_ik_setup{self.model}.xml")
                ik_solver = osim.InverseKinematicsTool(trial_ik_setup)
            ik_solver.set_marker_file(marker_file_path)
            ik_solver.set_results_directory(self.ik_dir)
            ik_solver.set_output_motion_file(output_motion_file_path)
            ik_solver.run()

            # The file sink is global, so remove it after every run
            osim.Logger.removeFileSink()

        # Filter the IK output to improve ID results
        if filter_output:
            output_header, output_data_df = read_mot_sto(output_motion_file_path)
            output_data_arr = output_data_df.to_numpy()
            output_data_filtered = Filter.butter_filt(output_data_arr[:, 1:], cutoff_freq, self.freq_marker, axis=0)
            output_data_filtered_df = pd.DataFrame(np.column_stack([output_data_arr[:, 0], output_data_filtered]), columns=output_data_df.columns)
            save_mot(output_motion_file_path, output_header, output_data_filtered_df)

        return output_motion_file_path

    def solve_id(self, force_file_path, overwrite):
        """
        Solve inverse dynamics of the trial by calling OpenSim API.
        
        :param force_file_path: The path of the file containing both GRF and boxing pad data. The data should be in .mot or .sto format.
        :param overwrite: Whether to overwrite the existing ID output file.
        """
        print(f"Start solving ID for trial {self.trial_idx} at {self.speed} speed")
        if not os.path.exists(self.id_dir):
            os.makedirs(self.id_dir)
        output_gen_force_file_path = os.path.join(self.id_dir, os.path.basename(self.id_dir) + '_id_generalized_forces.sto')
        if os.path.exists(output_gen_force_file_path) and not overwrite:
            print(f"ID output file {output_gen_force_file_path} already exists. Load the existing file directly as overwrite is set to False.")
            return output_gen_force_file_path
        elif os.path.exists(output_gen_force_file_path):
            print(f"ID output file {output_gen_force_file_path} already exists. Overwrite the file as overwrite is set to True.")
        # osim.Logger.setLevelString('Warn')
        log_file_path = os.path.join(self.id_dir, 'opensim.log')

        # Temporarily change the working directory to the trial's ID directory to be compatible with the relative path 
        # in the setup file, and suppress the output of the OpenSim API to accelerate the process.
        with temp_chdir(self.subject.subject_id_dir), suppress_cpp_stdout(log_file_path):
            osim.Logger.removeFileSink()
            osim.Logger.addFileSink(log_file_path)
            osim.Logger.setLevelString('Info')

            # if not hasattr(self.subject, 'id_solver') or not self.subject.id_solver:
            if not self.model:
                id_solver = osim.InverseDynamicsTool(self.subject.subject_id_setup)
            else:
                trial_id_setup = os.path.join(self.subject.subject_id_dir, self.subject.subject_id + f"_id_setup{self.model}.xml")
                id_solver = osim.InverseDynamicsTool(trial_id_setup)
            if hasattr(self, 'ik_results'):
                coordinate_file_path = self.ik_results
            else:
                coordinate_file_path = os.path.join(self.ik_dir, os.path.basename(self.ik_dir) + '_ik_output.mot')
            id_solver.setCoordinatesFileName(coordinate_file_path)
            id_solver.set_results_directory(self.id_dir)
            id_solver.setOutputGenForceFileName(os.path.split(output_gen_force_file_path)[1])
            
            # Explicitly export external loads to .xml file after writing data file name, then reload the .xml, 
            # because the InverseDynamicsTool can't set data file name for some reason
            external_loads_xml_file = id_solver.getExternalLoadsFileName()
            temp_loads_xml_file = self._update_external_loads_datafile(external_loads_xml_file, force_file_path, "id_external_loads_temp.xml")
            id_solver.setExternalLoadsFileName(temp_loads_xml_file)

            id_solver.run()
            id_solver.setExternalLoadsFileName(external_loads_xml_file)
            os.remove(temp_loads_xml_file)
        return output_gen_force_file_path
    
    def solve_ik_solver(self, marker_file_path, overwrite, filter_output=True, cutoff_freq=10):
        """
        High-performance IK solve using InverseKinematicsSolver instead of the IK Tool.
        Bypasses Tool-level I/O overhead and caches the model/solver in memory.
        """
        def _get_reference(ik_taskset: osim.IKTaskSet):
            """Convert IKTaskSet to MarkersReference and CoordinateReference"""
            # Extract the names and weights from the tasks
            m_ref = osim.MarkersReference()
            coord_refs = osim.SimTKArrayCoordinateReference()
            marker_weights = osim.SetMarkerWeights()
            
            # Create Set of MarkerWeight
            for i in range(ik_taskset.getSize()):
                task = ik_taskset.get(i)
                if osim.IKMarkerTask.safeDownCast(task):
                    marker_task = osim.IKMarkerTask.safeDownCast(task)
                    if marker_task.getApply():
                        # marker_names.append(marker_task.getName())
                        marker_weights.cloneAndAppend(osim.MarkerWeight(marker_task.getName(), marker_task.getWeight()))
                elif osim.IKCoordinateTask.safeDownCast(task):
                    coord_task = osim.IKCoordinateTask.safeDownCast(task)
                    if coord_task.getApply():
                        cr = osim.CoordinateReference(coord_task.getName(), osim.Constant(0)) 
                        cr.setWeight(coord_task.getWeight())
                        coord_refs.push_back(cr)

            # Create the reference object the solver actually wants
            # Pass empty marker positions for now; they get updated in the loop
            try:
                units = model.getLengthUnits()
                m_ref = osim.MarkersReference(marker_file_path, marker_weights, units)
            except:
                m_ref.set_marker_file(marker_file_path)
                m_ref.set_marker_weights(marker_weights)
            
            return m_ref, coord_refs
        
        print(f"Start optimized IK for trial {self.trial_idx} at {self.speed} speed")
        if not os.path.exists(self.ik_dir):
            os.makedirs(self.ik_dir)
        output_motion_file_path = os.path.join(self.ik_dir, os.path.basename(self.ik_dir) + '_ik_output.mot')
        
        if os.path.exists(output_motion_file_path) and not overwrite:
            print(f"IK output file {output_motion_file_path} already exists. Load the existing file directly as overwrite is set to False.")
            return output_motion_file_path
        elif os.path.exists(output_motion_file_path):
            print(f"IK output file {output_motion_file_path} already exists. Overwrite the file as overwrite is set to True.")

        with temp_chdir(self.subject.subject_ik_dir), suppress_cpp_stdout():
            # 1. SETUP CACHED MODEL AND SOLVER
            # Store the solver/model in the subject object to avoid re-loading every trial
            # if not hasattr(self.subject, 'ik_solver') or not isinstance(self.subject.ik_solver, osim.InverseKinematicsSolver):
            # Load Setup/Tasks from existing XML
            ik_tool_template = osim.InverseKinematicsTool(self.subject.subject_ik_setup)
            task_set = ik_tool_template.getIKTaskSet()
            accuracy = ik_tool_template.get_accuracy()
            
            # Load model
            model_path = ik_tool_template.get_model_file()
            model = osim.Model(model_path)
            # model.set_length_units("Meters")

            # Initialize Solver
            markers_reference, coord_references = _get_reference(task_set)
            constraint_weight = ik_tool_template.get_constraint_weight()
            if abs(constraint_weight) == math.inf:
                self.subject.ik_solver = osim.InverseKinematicsSolver(model, markers_reference, coord_references)
            else:
                self.subject.ik_solver = osim.InverseKinematicsSolver(model, markers_reference, coord_references, round(constraint_weight))
            self.subject.ik_solver.setAccuracy(accuracy)
            state = model.initSystem()
            
            # 2. LOAD EXPERIMENTAL DATA
            # Load TRC directly into memory
            marker_table = osim.TimeSeriesTableVec3(marker_file_path)
            time_column = marker_table.getIndependentColumn()
            n_frames = marker_table.getNumRows()
            n_coords = model.getNumCoordinates()
            # Prepare storage for results
            coord_names = [model.getCoordinateSet().get(i).getName() for i in range(n_coords)]
            ik_results = np.zeros((n_frames, n_coords))

            # 3. SOLVE FRAME-BY-FRAME
            # This loop is significantly faster than Tool.run() because it avoids file I/O per frame
            for i in range(n_frames):
                # Update solver with current frame's marker positions
                current_time = time_column[i]
                state.setTime(current_time)
                self.subject.ik_solver.assemble(state)
                
                # Extract coordinate values (converting to degrees for .mot compatibility if needed)
                # OpenSim standard .mot files usually store in Degrees, but solver works in Radians
                # MotionType: 0 - Undefined, 1 - Rotational, 2 - Translational, 3 - Coupled
                for j in range(n_coords):
                    coord = model.getCoordinateSet().get(j)
                    ik_results[i, j] = coord.getValue(state) * (180.0 / np.pi if coord.getMotionType() == 1 else 1.0)

            # 4. CONVERT TO DATAFRAME & SAVE
            if filter_output:
                ik_results = Filter.butter_filt(ik_results, cutoff_freq, self.freq_marker, axis=0)
            output_data_df = pd.DataFrame(np.column_stack([time_column, ik_results]), columns=['time']+coord_names)
            output_header = '\n'.join(["Coordinates", "version=1", f"nRows={n_frames}", f"nColumns={n_coords+1}", "inDegrees=yes", "endheader\n"])
            save_mot(output_motion_file_path, output_header, output_data_df)

        return output_motion_file_path

    def summarize_trial(self, idx_event_cropped, inspect_moment):
        """
        Summarize the analysis result of a single trial.
        """
        # Extract trial info
        # Extract velocity
        self.velocity, COM_trajectory = self.get_velocity(self.data_normalized['markers'])
        for i, event in enumerate(idx_event_cropped['markers']):
            if event[0] == 'Anchor Step':
                idx_anchor_step = event[1]
                # The mean speed 1 stride before anchor step
                mean_speed_before_anchor_x = np.mean(self.velocity[0, idx_event_cropped['markers'][i-2][1]:idx_anchor_step])
                mean_speed_before_anchor_y = np.mean(self.velocity[1, idx_event_cropped['markers'][i-2][1]:idx_anchor_step])
                mean_speed_before_anchor_z = np.mean(self.velocity[2, idx_event_cropped['markers'][i-2][1]:idx_anchor_step])
                mean_speed_before_anchor_total = np.mean(self.velocity[3, idx_event_cropped['markers'][i-2][1]:idx_anchor_step])
                mean_y_before_anchor = np.mean(COM_trajectory[1, :])
        # Extract push info
        self.pad_force_total = get_vector_norm(self.data_normalized['pad']['force'])
        push_pad_ind = extract_push_range(self.pad_force_total, self.freq_pad) \
            if self.intensity != 'fake' else [math.nan, math.nan, math.nan]
        push_marker_ind = get_nearest_index(push_pad_ind, self.freq_pad, self.freq_marker)
        # Extract ID results
        _, trial_id_results = read_mot_sto(self.id_results)
        side_abbr = self.subject.dominant_hand[0].lower()
        hip_add_moment = trial_id_results[f'hip_adduction_{side_abbr}_moment'].to_numpy(copy=True)
        knee_add_moment = trial_id_results[f'knee_add_{side_abbr}_moment'].to_numpy(copy=True)
        ankle_invers_moment = trial_id_results[f'subtalar_angle_{side_abbr}_moment'].to_numpy(copy=True)
        hip_response_idx, hip_response_moment = find_response_moment(-hip_add_moment, idx_event_cropped['markers'][0][1], idx_anchor_step, idx_event_cropped['markers'][-1][1])
        knee_response_idx, knee_response_moment = find_response_moment(-knee_add_moment, idx_event_cropped['markers'][0][1], idx_anchor_step, idx_event_cropped['markers'][-1][1])
        ankle_response_idx, ankle_response_moment = find_response_moment(-ankle_invers_moment, idx_event_cropped['markers'][0][1], idx_anchor_step, idx_event_cropped['markers'][-1][1])
        # Create output dict and dataframe
        trial_extract_dict = {
            "subject_id": self.subject.subject_id,
            "subject_weight": self.subject.weight,
            "subject_height": self.subject.height,
            "total_trial_idx": self.total_trial_idx,
            "speed": self.speed,
            "trial_idx": self.trial_idx,
            "intensity": self.intensity,
            "anchor_step_time": (idx_anchor_step + 1) / self.freq_marker,
            "mean_speed_before_anchor": mean_speed_before_anchor_total,
            "push_start_time": (push_pad_ind[0] + 1) / self.freq_pad,
            "push_peak_time": (push_pad_ind[1] + 1) / self.freq_pad,
            "push_end_time": (push_pad_ind[2] + 1) / self.freq_pad,
            "push_duration": (push_pad_ind[2] - push_pad_ind[0] + 1) / self.freq_pad,
            "push_peak_force": self.pad_force_total[push_pad_ind[1]] if self.intensity != 'fake' else math.nan,
            "push_impulse": get_vector_norm(integrate_interval(self.data_normalized['pad']['force'], push_pad_ind[0], push_pad_ind[2], self.freq_pad)),  # Impulse is a vector
            "push_peak_force_norm": None,
            "push_impulse_norm": None,
            "delta_y_push": COM_trajectory[1, push_marker_ind[2]] - COM_trajectory[1, push_marker_ind[0]] if self.intensity != 'fake' else math.nan,
            "delta_y_post_anchor": min(COM_trajectory[1, idx_anchor_step:idx_event_cropped['markers'][-1][1]]) - mean_y_before_anchor,
            "delta_speed_push_x": self.velocity[0, push_marker_ind[2]] - self.velocity[0, push_marker_ind[0]] if self.intensity != 'fake' else math.nan,
            "delta_speed_push_y": self.velocity[1, push_marker_ind[2]] - self.velocity[1, push_marker_ind[0]] if self.intensity != 'fake' else math.nan,
            "delta_speed_push_z": self.velocity[2, push_marker_ind[2]] - self.velocity[2, push_marker_ind[0]] if self.intensity != 'fake' else math.nan,
            # "delta_speed_push_total": self.velocity[3, push_marker_ind[2]] - self.velocity[3, push_marker_ind[0]] if self.intensity != 'fake' else math.nan,
            "delta_speed_post_anchor_x": min(self.velocity[0, idx_anchor_step:idx_event_cropped['markers'][-1][1]]) - mean_speed_before_anchor_x if side_abbr == 'r'\
                else max(self.velocity[0, idx_anchor_step:idx_event_cropped['markers'][-1][1]]) - mean_speed_before_anchor_x,
            "delta_speed_post_anchor_y": min(self.velocity[1, idx_anchor_step:idx_event_cropped['markers'][-1][1]]) - mean_speed_before_anchor_y,
            "delta_speed_post_anchor_z": max(self.velocity[2, idx_anchor_step:idx_event_cropped['markers'][-1][1]]) - mean_speed_before_anchor_z,
            # "delta_speed_post_anchor_total": max(self.velocity[3, idx_anchor_step:idx_event_cropped['markers'][-1][1]]) - mean_speed_before_anchor_total,
            "delta_hip_add_moment_push_norm": (hip_add_moment[push_marker_ind[2]] - hip_add_moment[push_marker_ind[0]]) / (self.subject.weight * self.subject.height)\
                if self.intensity != 'fake' else math.nan,
            "delta_knee_add_moment_push_norm": (knee_add_moment[push_marker_ind[2]] - knee_add_moment[push_marker_ind[0]]) / (self.subject.weight * self.subject.height)\
                if self.intensity != 'fake' else math.nan,
            "delta_ankle_invers_moment_push_norm": (ankle_invers_moment[push_marker_ind[2]] - ankle_invers_moment[push_marker_ind[0]]) / (self.subject.weight * self.subject.height)\
                if self.intensity != 'fake' else math.nan,
            "hip_response_moment_post_anchor_norm": hip_response_moment / (self.subject.weight * self.subject.height),
            "knee_response_moment_post_anchor_norm": knee_response_moment / (self.subject.weight * self.subject.height),
            "ankle_response_moment_post_anchor_norm": ankle_response_moment / (self.subject.weight * self.subject.height),
            "hip_response_moment_time": (hip_response_idx + 1) / self.freq_marker,
            "knee_response_moment_time": (knee_response_idx + 1) / self.freq_marker,
            "ankle_response_moment_time": (ankle_response_idx + 1) / self.freq_marker 
        }
        # Normalize the force and impulse by body mass
        trial_extract_dict['push_peak_force_norm'] = trial_extract_dict['push_peak_force'] / self.subject.weight,
        trial_extract_dict['push_impulse_norm'] = trial_extract_dict['push_impulse'] / self.subject.weight
        # Flip the data of left-handed subjects to be in the same walking direction as right-handed ones
        if self.subject.dominant_hand.lower() == 'left':
            trial_extract_dict['delta_speed_push_x'] *= -1
            trial_extract_dict['delta_speed_post_anchor_x'] *= -1
        
        if inspect_moment:
            fig, ax = plt.subplots(1, 3, figsize=(12, 7.5), sharex=True)
            ax[0].plot(hip_add_moment)
            ax[0].set_ylabel('Hip Adduction Moment [N*m]')
            ax[0].vlines(hip_response_idx, min(hip_add_moment), max(hip_add_moment))
            ax[0].vlines(idx_anchor_step, min(hip_add_moment), max(hip_add_moment), colors='black')
            ax[0].vlines(push_marker_ind, min(hip_add_moment), max(hip_add_moment), linestyles='dashed')
            ax[1].plot(knee_add_moment)
            ax[1].set_ylabel('Knee Adduction Moment [N*m]')
            ax[1].vlines(knee_response_idx, min(knee_add_moment), max(knee_add_moment))
            ax[1].vlines(idx_anchor_step, min(knee_add_moment), max(knee_add_moment), colors='black')
            ax[1].vlines(push_marker_ind, min(knee_add_moment), max(knee_add_moment), linestyles='dashed')
            ax[2].plot(ankle_invers_moment)
            ax[2].set_ylabel('Ankle Inversion Moment [N*m]')
            ax[2].vlines(ankle_response_idx, min(ankle_invers_moment), max(ankle_invers_moment))
            ax[2].vlines(idx_anchor_step, min(ankle_invers_moment), max(ankle_invers_moment), colors='black')
            ax[2].vlines(push_marker_ind, min(ankle_invers_moment), max(ankle_invers_moment), linestyles='dashed')
            plt.suptitle(f'Total Trial Idx: {self.total_trial_idx}\nSpeed: {self.speed}  Trial Idx: {self.trial_idx}  Intensity: {self.intensity}')
            plots_dir = os.path.join(ANALYSIS_DIR, r"Stats\Plots\Response Moment")
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            # plt.show()
            fig.savefig(os.path.join(plots_dir, f'{self.subject.subject_id}_{self.speed.capitalize()}_Trial{self.trial_idx}.png'), dpi=200, bbox_inches='tight')
            plt.close()

        trial_extract_df = pd.DataFrame(trial_extract_dict, index=[0])
        
        return trial_extract_df

    def _crop_by_events(self, data, freq, start_step_offset=2, end_step_offset=1, axis=-1):
        """
        Crop the data by ground contact events.

        :param data: The data to be cropped. Should be numpy.ndarray or dict.
        :param freq: Sampling frequency of the input data.
        :param start_step_offset: The number of step(s) before anchor step to start cropping.
        :param end_step_offset: The number of step(s) after anchor step to end cropping.
        :param axis: The axis to crop the data along (i.e., time axis).
        """
        # The offset to account for the difference between the pre-cropped data and the full trial measurement
        # In other words, the number of the first frame of input data in the full trial data
        frame_offset = math.floor((self.mocap_metadata['start_frame'] - 1) / self.freq_marker * freq)

        event_list_full = self.mocap_metadata['events']['labels']
        event_times_full = self.mocap_metadata['events']['times']
        idx_anchor_step = event_list_full.index('Anchor Step')

        # Set cropping frame range corresponding to the reference (marker) data, expressed in full trial data
        idx_start_frame_full_ref = round(event_times_full[idx_anchor_step - start_step_offset] * self.freq_marker)
        idx_end_frame_full_ref = round(event_times_full[idx_anchor_step + end_step_offset] * self.freq_marker)

        # Get corresponding cropping frame range of the input data source, expressed in full trial data
        idx_start_frame_full = math.floor((idx_start_frame_full_ref - 1) / self.freq_marker * freq) + 1
        idx_end_frame_full = math.ceil(idx_end_frame_full_ref / self.freq_marker * freq)
        # idx_start_frame_full = math.ceil(event_times_full[idx_anchor_step - start_step_offset] * freq)
        # idx_end_frame_full = math.ceil(event_times_full[idx_anchor_step + end_step_offset] * freq)
        
        # Get the corresponding cropping frame range in the input data
        # Note: The indices of frame here start from 0, and those above start from 1
        idx_start_frame = idx_start_frame_full - frame_offset
        idx_end_frame = idx_end_frame_full - frame_offset

        # Create dict for storing the indices of contact event frames in the output data
        idx_steps = np.arange(idx_anchor_step - start_step_offset, idx_anchor_step + end_step_offset + 1)
        idx_events = []
        for i in idx_steps:
            event_name = event_list_full[i]
            event_time = event_times_full[i]
            idx_event_frame = math.ceil(round(event_time * self.freq_marker) / self.freq_marker * freq) - idx_start_frame_full
            idx_events.append([event_name, idx_event_frame])

        if isinstance(data, np.ndarray):
            return np.take(data, np.arange(idx_start_frame, idx_end_frame + 1), axis=axis), idx_events
        elif isinstance(data, dict):
            return {field: np.take(field_data, np.arange(idx_start_frame, idx_end_frame + 1), axis=axis)
                    for field, field_data in data.items()}, idx_events
    
    def _update_external_loads_datafile(self, input_xml_path, datafile_path, output_xml_path=''):
        """
        Update the input xml file by adding a data file path to the `<datafile>` tag.
        """
        # Parse the XML file
        tree = ET.parse(input_xml_path)
        root = tree.getroot()

        # Find the <datafile> tag
        datafile_tag = root.find(".//datafile")

        if datafile_tag is not None:
            # Assign the new file path
            datafile_tag.text = datafile_path
        else:
            # If missing, create the <datafile> tag under the root (ExternalLoads)
            datafile_tag = ET.SubElement(root, "datafile")
            datafile_tag.text = datafile_path
        
        # Save the changes
        # Use xml_declaration=True to keep the header
        if output_xml_path:
            tree.write(output_xml_path, encoding="UTF-8", xml_declaration=True)
            return output_xml_path
        else:
            tree.write(input_xml_path, encoding="UTF-8", xml_declaration=True)
            return input_xml_path


def detect_ground_contact(self, data, criterium: str = 'marker'):
    """
    Identify the frame when a ground contact happens.
    
    :param data: The input data.
    :param criterium: Which data to use as the criterium for detection of ground contact of the non-dominant foot preceding the target step. Can be 'marker', 'grf', or None. \n
        If 'marker', the heel marker's vertical position is used to identify ground contact. \n
        If 'grf', the vertical GRF is used to identify ground contact. \n
        If None, no normalization is performed and the original data is returned.
    """
    pass