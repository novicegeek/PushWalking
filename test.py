import os
import matplotlib.pyplot as plt
from file import read_pad_data


test_dir = r"D:\Overseas\German Sport University Cologne\20.Course Materials\TSM11-Project&Applied Research Methods\20.Validation and Test Measurement\Test\Pad Orientation_20260318"
files = os.listdir(test_dir)
for i, file in enumerate(sorted(files)):
    if 'Long' in file:
        continue
    deg = file.split('_')[1]
    file_path = os.path.join(test_dir, file)
    data = read_pad_data(file_path, None, None, 5000, transform_to_global=False, spectrum_subtract=False, silence=False)
    fig, axes = plt.subplots(3, 1, sharex=True)
    for j, field in enumerate(['force', 'moment', 'center_of_pressure']):
        ax = axes[j]
        ax.set_prop_cycle(color=['red', 'yellow', 'blue'])
        ax.plot(data[field].T)  
    fig.legend(['X', 'Y', 'Z'])
    plt.title(f"Press at {deg}")
    plt.show(block=False)
plt.show()