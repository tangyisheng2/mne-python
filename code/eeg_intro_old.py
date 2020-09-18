"""
@File    :   eeg_intro
@Contact :   tangyisheng2@sina.com
@License :   (C)Copyright 1999-2020, Tang Yisheng

@Modify Time        @Author     @Version        @Description
------------        -------     --------        -----------
2020/8/11     tangyisheng2        1.0             Release
"""

import logging

import numpy as np
import pandas as pd

import mne

logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)

# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_folder = /EEG/EEGData/Emotiv/
# sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis_raw.fif')
# raw = mne.io.read_raw_edff(sample_data_raw_file, verbose=False).crop(tmax=120)
raw = mne.io.read_raw_edf("../EEGData/pain(1).edf", verbose=True)
raw.load_data()

# Print Raw data Info
print(raw)
print("EEG Info:\n" + str(raw.info))
print("EEG Channel Type:\n" + str(raw.get_channel_types()))
print("EEG Channel Name:\n" + str(raw.ch_names))

# Print raw data
raw_data = raw.get_data()
# data = pd.DataFrame(raw_data.T, columns=raw.ch_names)  # 转置数组
data = pd.DataFrame(raw_data.T, columns=raw.ch_names)  # 转置数组
# print(data["AF3"])
data = data.T  # ？在，为什么又转置回来

raw.pick_channels(['AF3', 'T7', 'Pz', 'T8', 'AF4', 'Status'])
raw.plot()

# Time slice

# create epoch (!Not working!)
# raw.copy().pick_types(eeg=True).plot(start=3, duration=6)
events = mne.find_events(raw)
# 第一列timestamp, 第三列为触发的类型


# Prepare for slice
time_stamp = data.loc["TIME_STAMP_s"] * 1000 + data.loc["TIME_STAMP_ms"]  # Create array for time_stamp in second
data_to_be_sliced = pd.DataFrame(raw.get_data(raw.ch_names), columns=time_stamp.T)
# REAL TIME =timestamp started (ms)+ TIME_STAMP_s*1000 + TIME_STAMP_ms
# 0.01s = 128 sample
sliced_instance_array = list()
sliced_instance_len = 2  # each instance in second
sliced_window_len = 1  # each window of instance in second
sliced_window = 128 * (sliced_window_len / 0.01)
for event_index in range(0, int((events.shape[1] - 1) / 2)):
    sliced_window_array = list()
    for window_index in range(0, int(sliced_instance_len / sliced_window_len)):
        sliced_window_array.append(data_to_be_sliced.iloc[:,
                                   int(events[event_index][0] + (event_index + sliced_window) * window_index):int(
                                       events[event_index][0] + (event_index + sliced_window) * (window_index + 1))])
    sliced_instance_array.append(sliced_window_array)

# Plot the PSD
raw.plot_psd()

# Calculate PSD
psd_list, freqs = raw.calculate_psd()

# Create structure for output
column = ["t-AF3", "t-T7", "t-Pz", "t-T8", "t-AF4",
          "a-AF3", "a-T7", "a-Pz", "a-T8", "a-AF4",
          "b-AF3", "b-T7", "b-Pz", "b-T8", "b-AF4"]
psd_mean = np.zeros([3, 5], dtype=float)

# Convert psd_list to matrix (Share:[Index of time windows, frequencies from 0-64Hz])
psd_matrix = np.asarray(psd_list)

# convert uV (times 10e9)
psd_matrix = psd_matrix * 10e6
# Convert frequency index to matrix(will work as columns in DataFrame)
freqs_matrix = np.asarray(freqs)

# Create Pandas DataFrame
psd_pd = pd.DataFrame(psd_matrix[0], columns=list(freqs))

# Create slice for different frequency bands
theta_slice = psd_pd.iloc[:, 64:128]
alpha_slice = psd_pd.iloc[:, 129:208]
beta_slice = psd_pd.iloc[:, 209:480]

# Calculate means
psd_mean[0, :] = theta_slice.mean(axis=1)
psd_mean[1, :] = alpha_slice.mean(axis=1)
psd_mean[2, :] = beta_slice.mean(axis=1)

# Reshape matrix into vector
psd_mean_reshape = psd_mean.reshape([1, 15])

# Visualize PSD vector
psd_mean_pd = pd.DataFrame(psd_mean_reshape, columns=column)
pass
