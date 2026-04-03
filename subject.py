import dataclasses
from dataclasses import dataclass
import os
import numpy as np
import opensim as osim
import pandas as pd
import traceback
from basic import VoltConverter
from file import read_mocap_data, read_pad_data, export_to_trc
from readc3d import C3DReader
from trial import Trial


DATA_DIR = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\22.Measurement Data"
ANALYSIS_DIR = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis"
STATS_DIR = os.path.join(ANALYSIS_DIR, "Stats")


@dataclass
class Subject():
    """
    Instance of a single subject.
    """
    subject_id: str
    date_of_measurement: str
    name: str
    sex: str
    date_of_birth: str
    age: int
    weight: float
    height: float
    dominant_hand: str
    model: dict = dataclasses.field(default_factory=dict)
    mocap_data_format: str = '.c3d'
    freq_pad: int = 5000
    n_trials: int = 72
    synced: bool = True
    subject_data_dir: str = dataclasses.field(init=False)
    subject_ik_dir: str = dataclasses.field(init=False)
    subject_ik_setup: str = dataclasses.field(init=False)
    ik_solver: osim.InverseKinematicsTool = dataclasses.field(init=False, default=None)
    subject_info: dict = dataclasses.field(init=False)
    subject_stats_dir: str = dataclasses.field(init=False)

    def __post_init__(self):
        self.subject_data_dir = os.path.join(DATA_DIR, '_'.join([self.subject_id, self.date_of_measurement]))
        self.subject_ik_dir = os.path.join(ANALYSIS_DIR, 'Inverse Kinematics', self.subject_id)
        if not self.model:
            self.subject_ik_setup = os.path.join(self.subject_ik_dir, self.subject_id + '_ik_setup.xml')
        else:
            self.subject_ik_setup = ''
        # Initialize IK solver (Update: Don't do this here, to prevent memory contamination)
        # self.ik_solver = osim.InverseKinematicsTool(self.subject_ik_setup)
        self.subject_id_dir = os.path.join(ANALYSIS_DIR, 'Inverse Dynamics', self.subject_id)
        if not self.model:
            self.subject_id_setup = os.path.join(self.subject_id_dir, self.subject_id + '_id_setup.xml')
        else:
            self.subject_id_setup = ''
        # Initialize ID solver (Update: Don't do this here, to prevent memory contamination)
        # self.id_solver = osim.InverseDynamicsTool(self.subject_id_setup)
        self.subject_stats_dir = os.path.join(STATS_DIR, self.subject_id)
        # Read trial list
        trial_list_name = '_'.join([self.subject_id, self.date_of_measurement, 'trial_list.csv'])
        trial_list_path = os.path.join(self.subject_data_dir, trial_list_name)
        self.trial_list = pd.read_csv(trial_list_path)

    def analyze_subject(self, to_read='all', **kwargs):
        """
        Analyze the data of the subject.
        """
        trial_extract_df_list = []
        for (total_trial_idx, speed, trial_idx, intensity, \
             c3d_data, c3d_metadata, c3d_file_path, pad_data, pad_file_path) in self.read_data(to_read=to_read):
            trial = Trial(mocap_data=c3d_data, mocap_metadata=c3d_metadata, mocap_file_path=c3d_file_path, 
                          pad_data=pad_data, pad_file_path=pad_file_path, 
                          total_trial_idx=total_trial_idx, speed=speed, trial_idx=trial_idx, intensity=intensity, subject=self)
            try:
                trial_extract_df = trial.analyze_trial(**kwargs)
                trial_extract_df_list.append(trial_extract_df)
            except Exception:
                log_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis"
                log_path = os.path.join(log_dir, self.subject_id + '_trial_error.log')
                with open(log_path, 'a') as f:
                    error_msg = traceback.format_exc()
                    msg = f"*** Error while analyzing trial {trial_idx} at {speed} speed for subject {self.subject_id} ***\n" \
                        f"[Error info]\n{error_msg}\n" + "*" * 50 + "\n"
                    print(msg)
                    f.write(msg)
                    f.close()
        return self.summarize_subject(trial_extract_df_list)

    def read_data(self, to_read='all'):
        """
        Read the measurement data of the subject from file, excluding static reference trial.
        """
        print("-" * 50)
        print(f"Reading data for subject {self.subject_id}")

        # Extract file paths list
        mocap_file_dir = os.path.join(self.subject_data_dir, 'Qualisys')
        pad_file_dir = os.path.join(self.subject_data_dir, 'Boxing Pad')
        
        # Initialize volt to force/moment converter
        volt_converter = VoltConverter(self.freq_pad)
        if self.mocap_data_format == '.c3d':
            c3dreader = C3DReader()
        else:
            c3dreader = None

        for i in range(self.trial_list.shape[0]):
            invalid_trials = int(self.trial_list.loc[i, 'Invalid Trials']) if not np.isnan(self.trial_list.loc[i, 'Invalid Trials']) else 0

            # Extract actual trial index, recorded trial index and speed of the trial
            total_trial_idx = int(self.trial_list.loc[i, 'Overall Trial No.'])
            trial_idx = int(self.trial_list.loc[i, 'Block Trial No.'])
            if trial_idx == 1:
                recorded_trial_idx = invalid_trials + 1
            else:
                recorded_trial_idx = int(self.trial_list.loc[i-1, 'Recorded Trial No.']) + invalid_trials + 1
            self.trial_list.loc[i, 'Recorded Trial No.'] = recorded_trial_idx
            speed = self.trial_list.loc[i, 'Block Speed']
            intensity = self.trial_list.loc[i, 'Intensity']

            if to_read != 'all' and total_trial_idx not in to_read:
                continue
            
            # Read MoCap file
            mocap_file = '_'.join([self.subject_id, self.date_of_measurement, speed.capitalize() + f"{int(recorded_trial_idx):04}"]) + self.mocap_data_format
            mocap_file_path = os.path.join(mocap_file_dir, mocap_file)
            mocap_data, mocap_metadata = read_mocap_data(mocap_file_path, self.mocap_data_format, c3dreader)
            
            # Read boxing pad file
            pad_file = '_'.join([self.subject_id, self.date_of_measurement, speed.capitalize() + f"{int(recorded_trial_idx):2}"]) + '.txt'
            pad_file_path = os.path.join(pad_file_dir, pad_file)
            pad_data = read_pad_data(pad_file_path, mocap_data, mocap_metadata, self.freq_pad, trial_idx, speed, volt_converter,
                                     spectrum_subtract=True, butterfilt=True)
                
            yield total_trial_idx, speed, trial_idx, intensity, mocap_data, mocap_metadata, mocap_file_path, pad_data, pad_file_path

        c3dreader.close()
    
    def _get_speed_trial_idx(self, filepath: str):
        """
        Get the corresponding walking speed for a particular measurement file.
        """
        speeds = ('slow', 'normal', 'fast', 'static')
        filename = os.path.split(filepath)[-1]
        speed_str = os.path.splitext(filename)[0].split('_')[2]
        for speed in speeds:
            if speed in speed_str.lower():
                trial_idx = int(speed_str[len(speed):])
                return speed, trial_idx
        raise ValueError(f"Cannot determine walking speed from filename: {filename}")
    
    def read_static_data(self, output=True):
        if self.mocap_data_format == '.c3d':
            c3dreader = C3DReader()
        else:
            c3dreader = None
        static_file_dir = os.path.join(self.subject_data_dir, 'Qualisys')
        for file in os.listdir(static_file_dir):
            if 'Static' not in file or os.path.splitext(file)[1] != self.mocap_data_format:
                continue
            static_file_path = os.path.join(static_file_dir, file)
            static_data, static_metadata = read_mocap_data(static_file_path, self.mocap_data_format, c3dreader)
            if output:
                output_path = os.path.splitext(static_file_path)[0] + '.trc'
                export_to_trc(static_data['markers'], static_metadata, output_path)
        c3dreader.close()

    def summarize_subject(self, trial_extract_df_list):
        """
        Summarize the analysis results of the subject and write to file.
        """
        subject_extract_df = pd.concat(trial_extract_df_list, ignore_index=True)
        if not os.path.exists(self.subject_stats_dir):
            os.makedirs(self.subject_stats_dir)
        subject_extract_df.to_csv(os.path.join(self.subject_stats_dir, self.subject_id + '_summary.csv'), index=False)
        return subject_extract_df