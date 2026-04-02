# Brief Intro to Each Script File

* *basic.py*

    **Core module** including classes and functions used for data processing and computation. `VoltConverter` is used for converting raw voltage data of the pad to forces and moments, `spectral_subtraction_same_trial`, `spectral_subtraction`, `Mask` and `Filter` are used for signal processing. `transform_pad_to_global` and `get_pad_center` are relevant to transforming the pushing data to global coordinate system.

* *descriptive_stats.py* and *descriptive_stats_subplots.py*

    Plotting modules. The first one plots individual figures, while the second one plots all figures as different subplots in a plot.

* *file.py*

    **Core module** for file handling.

* *main.py*
  
    Interface module to initiate the pipeline.

* *measuresheet.ipynb*

    Generate sheets used in the experiment, showing the order and condition of trials.

* *qtm_process.py*

    The automatic pipeline for processing marker data in QTM Python scripting interface.

* *readc3d.py*

    Read .c3d files in subprocess, as importing opensim and ezc3d modules in a same process leads to (seemingly) DLL conflicts.

* *readc3d_subprocess.py*

    The subprocess used to read .c3d files via ezc3d.

* *stats.py*

    **Core module** for statistics and visualization.

* *subject.py*

    **Core module** for building a `Subject` instance and processing for each subject.

* *test.py*

    For flexible testing purposes.

* *tmp.py*

    For temporary needs only.

* *trial.py*

    **Core module** for building a `Trial` instance and processing for each trial. 
