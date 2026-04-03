import qtm
import heapq
import os
os.chdir(r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods")
import math

# ==============================================================================
# 1. CONFIGURATION CLASS
# ==============================================================================

class QTMConfig:
    """
    Holds all user-modifiable settings.
    Access this in the terminal via: qtm.config.<setting>
    """
    def __init__(self, subject_id):
        # -- Processing Options --
        self.gap_fill = True        # Set to True to enable auto gap-filling
        self.export_c3d = True       # Set to True to export C3D
        self.save = True             # Set to True to save the QTM file
        self.overwrite = False        # If True, overwrites original. If False, adds suffix.

        # -- Analysis Parameters --
        self.anchor_foot = "Right"    # Options: "Right" or "Left"
        self.force_threshold = 20.0   # Newtons
        self.aim_dir = os.path.abspath(r"Push Walking\AIM models")
        self.aims = {"Pad": os.path.join(self.aim_dir, "Pad_AIM.qam"),
                     "Subject": None}

        if subject_id:
            for file in os.listdir(self.aim_dir):
                if file.endswith(".qam") and subject_id in file:
                    self.aims["Subject"] = os.path.join(self.aim_dir, file)
                    break
            if self.aims["Subject"] is None:
                print(f"Can't find AIM model file for subject {subject_id}.")
        
        # -- Marker & Plate Labels --
        self.marker_l_heel = "L_HeelTop"
        self.marker_r_heel = "R_HeelTop"
        self.timeline_window = 0.25
        # Note: Force plates are usually index 1 and 2. 
        # Logic: Right Anchor uses FP2, Left Anchor uses FP1.
        
       

# ==============================================================================
# 2. HELPER FUNCTIONS (MATH & QTM API)
# ==============================================================================

def fetch_series_data(series_type, series_id, total_frames=None):
    """
    Fetches data series (Force or Trajectory) as a standard Python list.
    series_type: 'force' or 'trajectory'
    The series_id is the internal series ID of QTM, not the conventional ID.
    Returns: List of Z-values (None for empty frames).
    """
    try:
        if series_type == 'force':
            # Usually returns [Fx, Fy, Fz, ...]. We want Z (index 2).
            series = [frame["force"][2] if frame else None for frame in qtm.data.series.force.get_samples(series_id)]        
        elif series_type == 'trajectory':
            # Returns [X, Y, Z, Res]. We want Z (index 2).
            series = [frame["position"][2] if frame else None for frame in qtm.data.series._3d.get_samples(series_id, {"start": 0, "end": total_frames-1})]
    except Exception as e:
        print(f"Data fetch warning ({series_type} series {series_id}): {e}")
    finally:
        data_out = series if 'series' in dir() else []
    return data_out

def calc_mean(values):
    """Calculates mean of valid numbers in a list."""
    valid = [v for v in values if (v is not None and not math.isnan(v))]
    return sum(valid) / len(valid) if valid else None

def find_local_minima(values, freq, window, apply_baseline=True):
    """
    Returns list of indices where local minima occur in relation to a specific width of neighbor.
    `window` specifies the half-width of the time window (in second) to define the local minimum.
    """
    indices = []
    if len(values) < 3: return indices

    # Use the average at static posture as baseline
    baseline_start_idx = int(0.1 * freq)
    baseline_end_idx = int(0.3 * freq)
    if not apply_baseline:
        baseline = math.inf
    else:
        baseline = calc_mean(values[baseline_start_idx:baseline_end_idx])
        if baseline is not None and not math.isnan(baseline):
            baseline = calc_mean(values[baseline_start_idx:baseline_end_idx]) * 2  # Multiply by 2 to allow for some fluctuations during baseline period
        else:  # If there are all empty values in the beginning of the trial, take the mean of the global smallest values
        #     len_baseline = baseline_end_idx - baseline_start_idx
        #     clean_values = [x for x in values if x]
        #     baseline = calc_mean(heapq.nsmallest(len_baseline, clean_values)) * 2
        # if baseline is None or math.isnan(baseline):
            baseline = math.inf
    neighbor_width = int(window * freq)

    for i in range(neighbor_width, len(values)-neighbor_width):
        v, v_prev, v_next = values[i], values[i-neighbor_width:i], values[i+1:i+neighbor_width+1]
        # If the neighborhood contains empty value, local minimum cannot be determined
        f = lambda l: any(x is None or math.isnan(x) for x in l)
        if f([v]) or f(v_prev) or f(v_next): continue
        # The local minimum needs to be smaller than baseline
        if v < min(v_prev) and v < min(v_next) and v < baseline:
            indices.append(i)
    return indices, baseline

# ==============================================================================
# 3. CORE LOGIC CLASS
# ==============================================================================

class QTMProcessor:
    def __init__(self, filepath=None):
        self.update_filepath(filepath, False)
    
    def update_filepath(self, filepath=None, keep_old_config=True):
        """
        Update file path, together with subject ID and AIM model.
        By default (`keep_old_config=True`), the previous configurations will not be modified.
        """
        if not filepath and qtm.file.is_open():
            self.filepath = qtm.file.get_path()
        elif not filepath:
            print("Please input valid file path.")
            self.filepath = None
        else:
            self.filepath = os.path.abspath(filepath)

        self.subject_id = os.path.split(self.filepath)[-1].split('_')[0] if self.filepath is not None else None # Extract subject ID from filename

        if not keep_old_config:
            self.config = QTMConfig(self.subject_id)
        else:
            for file in os.listdir(self.config.aim_dir):
                if file.endswith(".qam") and self.subject_id in file:
                    self.config.aims["Subject"] = os.path.join(self.config.aim_dir, file)
                    break
            if self.config.aims["Subject"] is None:
                print(f"Can't find AIM model file for subject {self.subject_id}.")
        
    def _calculate_timeline(self):
        """Internal method to determine start/stop frames."""
        cfg = self.config
        
        # 1. Setup based on Anchor Foot
        if cfg.anchor_foot.lower() == "right":
            fp_id = 2  # Standard: Right-dominant strikes FP2, left-dominant strikes FP1
            marker_pre = cfg.marker_l_heel # Previous step is Left
            marker_sub = cfg.marker_r_heel # Next step is Right
        elif cfg.anchor_foot.lower() == "left":
            fp_id = 1
            marker_pre = cfg.marker_r_heel
            marker_sub = cfg.marker_l_heel
        
        # 2. Get Data
        # Note: The conventional ID is different from the internal ID that QTM uses.
        fp_series_id = qtm.data.series.force.get_series_ids()[fp_id-1]
        print(f"Analyzing: Anchor={cfg.anchor_foot} (FP{fp_id}, internal ID {fp_series_id}), Pre={marker_pre}, Sub={marker_sub}")
        
        # Force Data
        force_z = fetch_series_data('force', fp_series_id)
        if not force_z:
            print(f"Error: No data found for Force Plate {fp_id} (internal ID {fp_series_id}).")
            return None

        # Marker Data
        id_left = qtm.data.object.trajectory.find_trajectory(cfg.marker_l_heel)
        id_right = qtm.data.object.trajectory.find_trajectory(cfg.marker_r_heel)
        # id_pre = qtm.data.object.trajectory.find_trajectory(marker_pre)
        # id_sub = qtm.data.object.trajectory.find_trajectory(marker_sub)
        
        if not id_left or not id_right:
        # if not id_pre or not id_sub:
            # print(f"Error: Heel marker ({', '.join([marker for i, marker in enumerate((marker_pre, marker_sub)) if not (id_pre, id_sub)[i]])}) not found in file parameters.")
            print(f"Error: Heel marker ({', '.join([marker for i, marker in enumerate((id_left, id_right)) if not (id_left, id_right)[i]])}) not found in file parameters.")
            return None
        
        # Number of total frames
        # total_frames = qtm.data.series._3d.get_sample_count(id_pre)
        # total_frames = qtm.data.series._3d.get_sample_count(id_left)
        total_frames = qtm.gui.timeline.get_frame_count()

        # Note: Trajectory Series usually requires ID.
        # traj_pre = fetch_series_data('trajectory', id_pre)
        # traj_sub = fetch_series_data('trajectory', id_sub)
        traj_left = fetch_series_data('trajectory', id_left, total_frames)
        traj_right = fetch_series_data('trajectory', id_right, total_frames)

        # 3. Force Offset (0.1s to 0.3s)
        fp_freq = qtm.data.series.force.get_frequency(fp_series_id)
        start_idx = int(0.1 * fp_freq)
        end_idx = int(0.3 * fp_freq)

        subset = force_z[start_idx:end_idx] if len(force_z) > end_idx else []
        offset = calc_mean(subset)
        if offset is None:
            offset = 0
            print(f"Can't acquire force data offset from frame {start_idx} to {end_idx}. "
                  "Offset set to 0.")
        
        # 4. Anchor Detection (> Threshold)
        anchor_frame_fp = None
        for i, val in enumerate(force_z):
            if val and (val - offset) > cfg.force_threshold:
                anchor_frame_fp = i + 1
                break
        
        if anchor_frame_fp is None:
            print(f"Anchor step (Force > {cfg.force_threshold}N) not detected.")
            anchor_frame_marker = None
        else:
            anchor_frame_marker = round((anchor_frame_fp - 1) / fp_freq * self.config.marker_freq) + 1
            print(f"Anchor step detected at force plate frame: {anchor_frame_fp}, corresponding to marker frame: {anchor_frame_marker}")

        # 5. Step Extraction (Minima)
        # minima_pre = find_local_minima(traj_pre, marker_freq, self.config.timeline_window)
        # minima_sub = find_local_minima(traj_sub, marker_freq, self.config.timeline_window)
        minima_left, baseline_left = find_local_minima(traj_left, self.config.marker_freq, self.config.timeline_window)
        minima_right, baseline_right = find_local_minima(traj_right, self.config.marker_freq, self.config.timeline_window)
        
        # The find_local_minima() function returns indices, add one to get frame number
        # And exclude the frames that are locally minima, but belong to the anchor step
        if anchor_frame_marker is not None:
            left_contacts = [m + 1 for m in minima_left if self._is_heel_strike(traj_left, m, "Left", total_frames) and (
                                (m < anchor_frame_marker - self.config.marker_freq * self.config.timeline_window / 2) or 
                                (m > anchor_frame_marker + self.config.marker_freq * self.config.timeline_window / 2)
                                )]
            right_contacts = [m + 1 for m in minima_right if self._is_heel_strike(traj_right, m, "Right", total_frames) and (
                                (m < anchor_frame_marker - self.config.marker_freq * self.config.timeline_window / 2) or 
                                (m > anchor_frame_marker + self.config.marker_freq * self.config.timeline_window / 2)
                                )]
        else:
            left_contacts = [m + 1 for m in minima_left]
            right_contacts = [m + 1 for m in minima_right]
        
        if cfg.anchor_foot.lower() == "right":
            minima_pre = left_contacts
            minima_sub = right_contacts
        elif cfg.anchor_foot.lower() == "left":
            minima_pre = right_contacts
            minima_sub = left_contacts
        
        if anchor_frame_marker is not None:
            pre_candidates = [m for m in minima_pre if m < anchor_frame_marker - self.config.marker_freq * self.config.timeline_window / 2]
            sub_candidates = [m for m in minima_sub if m > anchor_frame_marker + self.config.marker_freq * self.config.timeline_window]
        else:
            pre_candidates = []
            sub_candidates = []

        # Preceding: 3 steps before (Need 2nd most recent contra-lateral contact)
        # The criteria for anchor frame and preceding/subsequent steps are different (force and marker data respectively)
        if len(pre_candidates) < 2:
            print("Timeline Error: Could not find 3 steps before anchor. The first frame after buffer is returned.")
            start_trim = 1
        else:
            # Counting backward from the end, Index -1 is step -1, Index -2 is step -3
            start_trim = pre_candidates[-2]
        
        # Subsequent: 2 steps after (Need 1st ipsi-lateral contact after anchor)
        # Filter for minima AFTER anchor, sort Ascending
        if len(sub_candidates) < 1:
            print("Timeline Error: Could not find 2 steps after anchor. The last frame before buffer is returned.")
            end_trim = total_frames
        else:
            end_trim = sub_candidates[0]

        # 6. Apply Buffer (0.1s)
        buffer_frames = int(0.1 * self.config.marker_freq)
        final_start_frame = max(buffer_frames + 1, start_trim - buffer_frames)
        final_end_frame = min(total_frames - buffer_frames, end_trim + buffer_frames)
        
        return final_start_frame, anchor_frame_marker, final_end_frame, left_contacts, right_contacts

    def _is_heel_strike(self, heel_data, minimum_idx, side, total_frames):
        """
        Determine if a local minimum of heel marker height is a real heel strike.
        """
        if side.lower() == "right":
            ref_marker = "R_ToesTop"
        elif side.lower() == "left":
            ref_marker = "L_ToesTop"
        else:
            print(f"The input {side} is not a invalid value for parameter side.")
            return None
        
        try:
            ref_series_id = qtm.data.object.trajectory.find_trajectory(ref_marker)
            ref_data = [frame["position"][2] if frame else None for frame in qtm.data.series._3d.get_samples(ref_series_id, {"start": 0, "end": total_frames-1})]
            ref_start_idx = int(minimum_idx - self.config.marker_freq * self.config.timeline_window)
            ref_end_idx = int(minimum_idx - self.config.marker_freq * self.config.timeline_window / 2)
        except:
            return False
        
        try:
            if calc_mean(heel_data[ref_start_idx:ref_end_idx]) > calc_mean(ref_data[ref_start_idx:ref_end_idx]) and heel_data[minimum_idx] < ref_data[minimum_idx]:
                # Primary criterium: During the first half of the swing phase preceding the strike, the heel marker should be higher than the reference marker
                # and at the moment of heel strike, the heel marker should be lower than the reference marker
                print(f"Heel strike detected for {side} foot at frame {minimum_idx+1} using primary criterium")
                return True
        except:
            pass
        # Secondary criterium if primary criterium doesn't pass
        # During the last period of the swing phase preceding the strike, the height difference between the two markers should be much larger
        # than during the baseline period
        baseline_start_idx = int(0.1 * self.config.marker_freq)
        baseline_end_idx = int(0.3 * self.config.marker_freq)
        baseline_ref = calc_mean(ref_data[baseline_start_idx:baseline_end_idx])
        baseline_heel = calc_mean(heel_data[baseline_start_idx:baseline_end_idx])
        if not baseline_ref or not baseline_heel:  # If there are all empty values in the beginning of the trial in either of the baselines, take the mean of the global smallest values
            len_baseline = baseline_end_idx - baseline_start_idx
            clean_values_ref = [x for x in ref_data if x]
            clean_values_heel = [x for x in heel_data if x]
            baseline_ref = calc_mean(heapq.nsmallest(len_baseline, clean_values_ref))
            baseline_heel = calc_mean(heapq.nsmallest(len_baseline, clean_values_heel))
        if not baseline_ref or not baseline_heel:
            return False
        baseline_diff = baseline_ref - baseline_heel
        sec_ref_start_idx = int(minimum_idx - self.config.marker_freq * self.config.timeline_window / 4)
        sec_ref_end_idx = minimum_idx
        try:
            if calc_mean(ref_data[sec_ref_start_idx:sec_ref_end_idx]) - calc_mean(heel_data[sec_ref_start_idx:sec_ref_end_idx]) > 3 * baseline_diff:
                print(f"Heel strike detected for {side} foot at frame {minimum_idx+1} using secondary criterium")
                return True
        except:
            pass
        return False

    def process(self):
        """
        Main function to run the processing pipeline on the CURRENT active file.
        """
        if self.filepath is None:
            print("No file path is available. Please specify a file to be processed by calling "
                  f"{self.__class__.__name__}.update_filepath(filepath)")
            return
        try:
            self.config.marker_freq = qtm.gui.timeline.get_frequency()

            # -- Task 1: Timeline --
            timeline = self._calculate_timeline()
            if timeline:
                start_frame, anchor_frame, end_frame, left_contacts, right_contacts = timeline
                qtm.data.object.event.clear_events()
                print("Clear all existing events before adding new ones.")
                if left_contacts:
                    for contact in left_contacts:
                        qtm.data.object.event.add_event({"label": "Left Contact",
                                                         "time": (contact - 1)/self.config.marker_freq,
                                                         "color": 0x00ffff})
                    print(f"Event added for left heel contact at frames {', '.join(list(map(str, left_contacts)))}")
                if right_contacts:
                    for contact in right_contacts:
                        qtm.data.object.event.add_event({"label": "Right Contact",
                                                         "time": (contact - 1)/self.config.marker_freq,
                                                         "color": 0x0000ff})
                    print(f"Event added for right heel contact at frames {', '.join(list(map(str, right_contacts)))}")
                if anchor_frame:
                    qtm.data.object.event.add_event({"label": "Anchor Step",
                                                    "time": (anchor_frame - 1)/self.config.marker_freq,
                                                    "color": 0x000000})
                    print(f"Event added for anchor step at frame {anchor_frame}")
                qtm.gui.timeline.set_selected_range({"start": start_frame, "end": end_frame})
                print(f"Timeline Set between Frame: {start_frame} - {end_frame}")
            
            # -- Task 2: 3D Track
            track_3d_settings = qtm.settings.processing._3d.get_settings("measurement")
            track_3d_settings['auto_limit_ray_length'] = False
            qtm.processing.track_3d(track_3d_settings)
            
            # -- Task 3: Gap Fill --
            if self.config.gap_fill:
                max_gap = int(0.1 * self.config.marker_freq)
                qtm.processing.fill_gaps({"max_gap_length": max_gap,
                                          "fill_type": "polynomial"}) 
                print(f"Gaps filled with a maximum of {max_gap} frames using polynomial algorithm.")

            # -- Task 4: AIM --
            qtm.processing.apply_aim({"models": {aim_path: {"is_applied": True} for aim_path in self.config.aims.values() if aim_path}})
            print(f"{(chr(10) + '  -').join(['AIM applied:'] + [aim_path for aim_path in self.config.aims.values() if aim_path])}")
            
            # -- Task 4: Export/Save --
            # We need the filename to handle suffixes
            # Since QTM scripting variable for filename is inconsistent across versions,
            # we rely on the user having a file open with a valid path.
            # We can try to deduce it if needed, or assume 'Save' works on current.
            
            if self.config.export_c3d:
                # To export properly we need a name. 
                # If we don't have the path, we can't formulate the command easily.
                # Assuming user has saved the file at least once:
                qtm.settings.export.c3d.set_exclude_unidentified(True)
                qtm.settings.export.c3d.set_use_full_label(True)
                qtm.settings.export.c3d.set_length_units("mm")
                qtm.settings.processing.set_export_c3d("capture", True)
                qtm.settings.processing.set_export_c3d("batch", True)
                qtm.settings.processing.set_export_c3d("reprocess", True)
                print("C3D Export option set.")

            if self.config.save:
                if self.config.overwrite:
                    qtm.file.save()
                    print("File saved (Overwritten).")
                else:
                    writepath = '_Processed'.join(os.path.splitext(self.filepath))
                    qtm.file.save_as(writepath)
                    print(f"File saved as a new file: {writepath}).")

            print("--- Processing Complete ---")

        except Exception as e:
            print(f"Processing Exception: {e}")

    def process_batch(self, folder_path):
        """
        Runs the process() method on all .qtm files in the specified folder.
        Multiple folders can be passed as list.
        """
        if isinstance(folder_path, list):
            for folder in folder_path:
                self.process_batch(folder)

        if not os.path.isdir(folder_path):
            print(f"Invalid folder path: {folder_path}")
            return

        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".qtm")]
        print(f"Found {len(files)} files in {folder_path}")

        # Enforce 'Save' for batch
        original_save_setting = self.config.save
        self.config.save = True

        for f in files:
            full_path = os.path.join(folder_path, f)
            if "_Processed" in f: continue # Skip already processed

            print(f"\nProcessing: {f}...")
            try:
                # print(f"Load {full_path}")
                qtm.file.open(full_path)
                # Update attributes
                self.update_filepath(full_path)
                self.process()
                if qtm.file.is_open():
                    qtm.file.close()
                print("Close")
            except Exception as e:
                print(f"Failed to process {f}: {e}")

        # Restore setting
        self.config.save = original_save_setting
        print("\nBatch Processing Finished.")

# ==============================================================================
# 4. INITIALIZATION
# ==============================================================================

# Create the global instance
qtm_process = QTMProcessor()

print("QTM Processor Script Loaded.")
print("Configurations:")
for key, value in vars(qtm_process.config).items():
    print(f"  {key}: {value}")
print("Usage:")
print("  1. Modify Settings:  qtm_process.config.attr = value")
print("  2. Run Current:      qtm_process.process()")
print("  3. Run Batch:        qtm_process.process_batch(r'C:\\Path\\To\\Folder')")