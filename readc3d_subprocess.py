import pickle
import struct
import sys
import ezc3d
import numpy as np
from basic import fix_cop_outliers


def run_reader():
    # Use binary mode for stdout to send pickle data
    stdout_bin = sys.stdout.buffer

    for line in sys.stdin:
        input_path = line.strip()
        if not input_path:
            break
    
        try:
            c = ezc3d.c3d(input_path, extract_forceplat_data=True)
            
            # Marker data shape: 4xNxT (components XYZ + 1, number of markers, number of frames)
            # The first 3 rows in the first element are X, Y, Z
            marker_data = c['data']['points'][:3, :, :].copy()
            # Convert to values in meter
            if c['parameters']['POINT']['UNITS']['value'][0] == 'mm':
                marker_data /= 1000

            # Force plate data
            fp_data = {}
            for i in range(2):
                fp_data[f"fp{i+1}"] = {
                    "force": c['data']['platform'][i]['force'].copy(),
                    "center_of_pressure": c['data']['platform'][i]['center_of_pressure'].copy(),
                    "moment": c['data']['platform'][i]['moment'].copy()
                }
                # Convert to values in meter and Newton*meter
                if c['data']['platform'][i]['unit_position'] == 'mm':
                    fp_data[f"fp{i+1}"]["center_of_pressure"] /= 1000
                if c['data']['platform'][i]['unit_moment'] == 'Nmm':
                    fp_data[f"fp{i+1}"]["moment"] /= 1000
                # Some COP values are exceptionally large/small, potentially due to incorrect decoding
                # of invalid data by ezc3d. Set those large invalid COP values to 0 
                # invalid_mask = np.abs(fp_data[f"fp{i+1}"]["center_of_pressure"]) > 10
                # fp_data[f"fp{i+1}"]['center_of_pressure'][invalid_mask] = 0
                fp_data[f"fp{i+1}"]['center_of_pressure'] = fix_cop_outliers(
                    fp_data[f"fp{i+1}"]['center_of_pressure'], -10, 10, axis=1)
            
            metadata = {
                "marker_labels": c['parameters']['POINT']['LABELS']['value'],
                "events": {"labels": c['parameters']['EVENT']['LABELS']['value'],
                           "times": np.array([60, 1]) @ c['parameters']['EVENT']['TIMES']['value']} if 'EVENT' in c['parameters'].keys() else {},
                "marker_freq": c['parameters']['POINT']['RATE']['value'][0],
                "fp_freq": c['parameters']['ANALOG']['RATE']['value'][0],
                "start_frame": c['parameters']['PROCESSING']['Cropped Measurement Start Frame']['value'][0] 
                                if 'Cropped Measurement Start Frame' in c['parameters']['PROCESSING'].keys() else 1,
                "end_frame": c['parameters']['PROCESSING']['Cropped Measurement End Frame']['value'][0] 
                                if 'Cropped Measurement End Frame' in c['parameters']['PROCESSING'].keys() else c['parameters']['POINT']['ORIGINAL_LAST_FRAME']['value'][0],
                "original_end_frame": c['parameters']['POINT']['ORIGINAL_LAST_FRAME']['value'][0],
                "marker_unit": 'm',
                "original_marker_unit": c['parameters']['POINT']['UNITS']['value'][0],
                "force_plate_corners": c['parameters']['FORCE_PLATFORM']['CORNERS']['value']
            }
            if metadata['original_marker_unit'] == 'mm':
                metadata['force_plate_corners'] /= 1000

            response = {"data": {"markers": marker_data, **fp_data}, "metadata": metadata}
            
        except Exception as e:
            response = {"error": str(e)}

        # Serialize and send
        serialized = pickle.dumps(response, protocol=pickle.HIGHEST_PROTOCOL)
        # Send size header (Big-endian unsigned int) then the data
        stdout_bin.write(struct.pack('>I', len(serialized)))
        stdout_bin.write(serialized)
        stdout_bin.flush()


if __name__ == "__main__":
    run_reader()