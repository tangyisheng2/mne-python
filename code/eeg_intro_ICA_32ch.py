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
from mne.preprocessing import (ICA)

logger = logging.getLogger('mne')  # one selection here used across mne-python
logger.propagate = False  # don't propagate (in case of multiple imports)

'''
Config
'''
# sample_data_folder = mne.datasets.sample.data_path()
# sample_data_folder = /EEG/EEGData/Emotiv/
# sample_data_raw_file = os.path.join(sample_data_folder, 'MEG', 'sample',
#                                     'sample_audvis_raw.fif')
# raw = mne.io.read_raw_edff(sample_data_raw_file, verbose=False).crop(tmax=120)
# 文件为捏肚子，痛觉
raw = mne.io.read_raw_edf("../EEGData/matlab_pain_eeglab.edf", verbose=True)
channel_num = 32
'''
Config End
'''
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

raw.pick_channels(raw.ch_names[4:4 + channel_num] + ["Status"])
# raw.plot()
# raw.plot(duration=60, proj=True)
# Filter for ICA
raw.filter(1, None, fir_design='firwin')

# # ICA
# regexp = r'AF3|T7|Pz|T8|AF4'
# artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
# raw.plot(order=artifact_picks, n_channels=len(artifact_picks))
#
# ecg_evoked = create_eeg_epochs(raw).average()
# ecg_evoked.apply_baseline(baseline=(None, -0.2))
# ecg_evoked.plot_joint()

# Time slice
# create epoch
# raw.copy().pick_types(eeg=True).plot(start=3, duration=6)
events = mne.find_events(raw, initial_event=True)
# 第一列timestamp, 第三列为触发的类型

# Creating epochs
# event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
#               'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=0, tmax=5, baseline=(0, 0), preload=True)
print(epochs)
epochs.plot(n_epochs=10)

# Go ICA
ica = ICA(n_components=1, method='fastica').fit(epochs)  # todo: check define of n_components

# eeg_inds, scores = ica.find_bads_eeg(epochs, threshold='auto')
#
# ica.plot_components(ecg_inds)

###############################################################################
# Plot properties of ECG components:
# ica.plot_properties(epochs, picks=eeg_inds)

###############################################################################
# Plot the estimated source of detected ECG related components
# ica.plot_sources(raw, picks=eeg_inds)

# Prepare for slice
time_stamp = data.loc["TIME_STAMP_s"] * 1000 + data.loc["TIME_STAMP_ms"]  # Create array for time_stamp in second
data_to_be_sliced = pd.DataFrame(raw.get_data(raw.ch_names), columns=time_stamp.T)
# REAL TIME =timestamp started (ms)+ TIME_STAMP_s*1000 + TIME_STAMP_ms
# 0.01s = 128 sample
sliced_instance_array = list()
sliced_instance_len = 5  # each instance in second
sliced_window_len = 1  # each window of instance in second
sliced_window = 128 * (sliced_window_len / 0.01)
for event_index in range(0, epochs.events.shape[0]):
    sliced_window_array = list()
    for window_index in range(0, int(sliced_instance_len / sliced_window_len)):
        sliced_window_array.append(epochs[0].copy().crop(tmin=window_index, tmax=window_index + 1, include_tmax=True))
    sliced_instance_array.append(sliced_window_array)

# Plot the PSD
raw.plot_psd()

psd_mean_with_slice = np.zeros([0, 3 * channel_num])
# Create structure for output
# 'Cz', 'Fz', 'Fp1', 'F7', 'F3', 'FC1', 'C3', 'FC5', 'FT9', 'T7', 'CP5', 'CP1',
# 'P3', 'P7', 'PO9', 'O1', 'Pz', 'Oz', 'O2', 'PO10', 'P8', 'P4', 'CP2', 'CP6', 'T8', 'FT10', 'FC6', 'C4',
# 'FC2', 'F4', 'F8', 'Fp2'

# Auto generate column
column = list()
for channel_name in raw.ch_names[0:-1]:  # Delete "Status" channel
    column.append(f"{'t-'}{channel_name}")
    column.append(f"{'a-'}{channel_name}")
    column.append(f"{'b-'}{channel_name}")
psd_mean = np.zeros([3, channel_num], dtype=float)

for event_index in range(0, epochs.events.shape[0]):
    for window_index in range(0, int(sliced_instance_len / sliced_window_len)):
        sliced_instance_array[event_index][window_index].plot_psd()

        # Calculate PSD
        psd_list, freqs = sliced_instance_array[event_index][window_index].calculate_psd()

        # Convert psd_list to matrix (Share:[Index of time windows, frequencies from 0-64Hz])
        # psd_matrix = np.asarray(psd_list)

        # # convert uV (times 10e6)
        # psd_matrix = psd_matrix * 10e6
        psd_list = psd_list[0] * 10e6
        # Convert frequency index to matrix(will work as columns in DataFrame)
        freqs_matrix = np.asarray(freqs)

        # Create Pandas DataFrame
        psd_pd = pd.DataFrame(psd_list, columns=freqs)

        # Create slice for different frequency bands
        theta_slice = psd_pd.iloc[:, 4:8]
        alpha_slice = psd_pd.iloc[:, 8:13]
        beta_slice = psd_pd.iloc[:, 13:30]

        # Calculate means
        psd_mean[0, :] = theta_slice.mean(axis=1)
        psd_mean[1, :] = alpha_slice.mean(axis=1)
        psd_mean[2, :] = beta_slice.mean(axis=1)

        # Reshape matrix into vector
        psd_mean_reshape = psd_mean.reshape([1, psd_mean.size])

        # Visualize PSD vector
        psd_mean_pd = pd.DataFrame(psd_mean_reshape, columns=column)
        psd_mean_with_slice = np.r_[psd_mean_with_slice, psd_mean_pd]

psd_mean_pd_with_slice = pd.DataFrame(psd_mean_with_slice, columns=column)
pass
