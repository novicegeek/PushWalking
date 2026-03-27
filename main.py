from importlib import reload
import os
import numpy as np
import pandas as pd
from subject import Subject
from trial import Trial
import matplotlib.pyplot as plt
import time
import scipy.signal as signal


DATA_DIR = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\22.Measurement Data"
ANALYSIS_DIR = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis"
STATS_DIR = os.path.join(ANALYSIS_DIR, "Stats")


subjects = {
    "S01": {
        "date_of_measurement": '20260123',
        "name": '',
        "sex": 'female',
        "date_of_birth": '',
        "age": 29,
        "weight": 65,
        "height": 1.69,
        "dominant_hand": 'right',
        "reprocess_trials": [],
        "model": {}
    },
    "S02": {
        "date_of_measurement": '20260126',
        "name": '',
        "sex": 'female',
        "date_of_birth": '',
        "age": 26,
        "weight": 72.5,
        "height": 1.63,
        "dominant_hand": 'right',
        "reprocess_trials": [],
        "model": {}
    },
    "S03": {
        "date_of_measurement": '20260130',
        "name": '',
        "sex": 'male',
        "date_of_birth": '',
        "age": 29,
        "weight": 73,
        "height": 1.69,
        "dominant_hand": 'right',
        "reprocess_trials": [],
        "model": {}
    },
    "S05": {
        "date_of_measurement": '20260210',
        "name": '',
        "sex": 'female',
        "date_of_birth": '',
        "age": 31,
        "weight": 69,
        "height": 1.71,
        "dominant_hand": 'left',
        "reprocess_trials": [],
        "model": {i: 1 for i in np.arange(1, 25)} | {i: 2 for i in np.arange(25, 73)}
    }
}

subject_extract_df_list = []
for subject_id, subject_info in subjects.items():
    # if subject_id not in ('S03'):
    #     continue 
    subject = Subject(subject_id, subject_info['date_of_measurement'], subject_info['name'], subject_info['sex'], 
                      subject_info['date_of_birth'], subject_info['age'], subject_info['weight'], subject_info['height'], subject_info['dominant_hand'],
                      subject_info['model'])
    # subject.read_static_data()
    problem_trials = subject_info['reprocess_trials']
    subject_extract_df = subject.analyze_subject(to_read='all', pause_analyze=problem_trials,
                                                 overwrite=False, solve_ik=False, solve_id=False)
    subject_extract_df_list.append(subject_extract_df)

all_subjects_df = pd.concat(subject_extract_df_list, ignore_index=True)
all_subjects_df.to_csv(os.path.join(STATS_DIR, 'all_subjects_summary.csv'), index=False)

# for speed, trial_idx, intensity, c3d_data, c3d_metadata, c3d_file_path, pad_data, pad_file_path in subject.read_data():
#     trial = Trial(mocap_data=c3d_data, mocap_metadata=c3d_metadata, mocap_file_path=c3d_file_path, 
#                   pad_data=pad_data, pad_file_path=pad_file_path, 
#                   speed=speed, trial_idx=trial_idx, subject=subject)
#     try:
#         trial.analyze_trial(overwrite=False)
#     except Exception as e:
#         log_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\23.Analysis"
#         log_path = os.path.join(log_dir, 'trial_error.log')
#         with open(log_path, 'a') as f:
#             message = f"***Error while analyzing trial {trial_idx} at {speed} speed for subject {subject_id}***" \
#                 f"\t\tError info: \n{e}\n" + "*" * 50
#             print(message)
#             f.write(message)
#             f.close()
    # velocity = trial.velocity
    # velocity = np.vstack([velocity, np.sqrt(np.sum(np.square(velocity), axis=0))])  # Calculate the overall velocity
    # event_times = c3d_metadata['events']['times'] - c3d_metadata['start_frame'] / c3d_metadata['marker_freq']