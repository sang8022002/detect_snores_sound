from scipy.signal import butter, lfilter, freqz
from scipy.signal import spectrogram
from pathlib import Path
from scipy.signal import find_peaks
from pydub import AudioSegment

import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import numpy as np
import io
import pandas as pd

# def simple_moving_average(data, window_size):
#     kernel = np.ones(window_size) / window_size  # Tạo kernel với giá trị bằng nhau
#     sma = np.convolve(data, kernel, 'same')  # 'same' để giữ kích thước ban đầu
#     return sma


# Tạo bộ lọc

# def create_bandpass_filter(lowcut, highcut, fs, order=4):

#     nyq = fs/2
#     low = lowcut/nyq
#     high = highcut/nyq
#     b,a = butter(order,[low, high], btype='band')
#     return b,a

# def apply_filter(data, lowcut, highcut, fs, order=4):
#     b,a = create_bandpass_filter(lowcut,highcut,fs,order=order)
#     y=lfilter(b,a,data)
#     return y

# Đọc tệp MP3
#input = ('snore_cut.mp3')
input = ('snore_10s.mp3')
#input = ('child_breath.mp3')
#input = ('heavy_breath.mp3')
name = Path(input).stem

audio = AudioSegment.from_mp3(input)
samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
print("Channels:", audio.channels)
print("Sample width:", audio.sample_width)
print("Frame rate:", audio.frame_rate)
print("Frame count:", len(audio))
sample_rate = audio.frame_rate
data = np.array(audio.get_array_of_samples())
time = np.arange(0, len(data)) / (sample_rate*2)

# Tạo bộ lọc thông dải
fs = audio.frame_rate  # Tần số lấy mẫu
#fs=20000
lowcut = 300
highcut = 2000
# data_ft=apply_filter(data,lowcut,highcut,fs,order=6)

#Đồ thị âm thanh trước khi lọc



#Đồ thi sau khi lọc
#MA code
# window_size = 1000
# sma_samples = simple_moving_average(data_ft, window_size)

# interval_2_5s = int(5 * sample_rate)
# interval_3s = int(4 * sample_rate)

# peaks, _ = find_peaks(data_ft, distance=interval_3s)
# max_peak_value = np.max(data_ft)
# threshold = max_peak_value * 1 / 3
# peaks_ft = [peak for peak in peaks if data_ft[peak] >= threshold]

# valid_peaks = []

# for peak in peaks_ft:
#     start = max(0, peak - interval_2_5s)
#     end = min(len(data_ft), peak + interval_2_5s)
#     if data_ft[peak] == max(data_ft[start:end]):
#         valid_peaks.append(peak)

# print(f"{len(peaks_ft)}")

# peak_distances = np.diff(peaks_ft)

# count=0
# for i in range(1,len(peak_distances)):
#     if peak_distances[i] > 2.5 * peak_distances[i - 1]:  # Kiểm tra điều kiện
#      count += 1  # Nếu đúng, tăng biến đếm

# print(f"{count}")

# analytic_signal = signal.hilbert(sma_samples)  # Tạo tín hiệu giải tích
# envelope = np.abs(analytic_signal)


# start_time = 1  # Start time in seconds
# end_time = 6  # End time in seconds

# start_index = int(start_time * sample_rate)  # Convert start time to index
# end_index = int(end_time * sample_rate)  # Convert end time to index
# extracted_data = data[start_index:end_index]  # Extract the range of data


derivative = np.gradient(data)
derivative = abs(derivative)
average_derivative = np.mean(derivative)
print(f"Average derivative: {average_derivative}")
derivative1 = derivative - average_derivative
derivative1[derivative1 > 0] = 1
derivative1[derivative1 <= 0] = 0
# print(f"Average derivative: {average_derivative}")
# derivative2 = np.gradient(derivative)
# plt.figure("Đạo hàm của tín hiệu data")
# plt.plot(time, derivative)
# plt.xlabel('Thời gian (s)')
# plt.ylabel('Đạo hàm')
# plt.title('Đạo hàm của tín hiệu data')

fig, axs = plt.subplots(3, 1, sharex = True)
axs[0].plot(time, data)
# for value in ampl_ppg_data_filtered:
#     axs[0].plot(value, median_data[value], "r*")
axs[0].set_xlabel("Thời gian (s)")
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Detect snore sound")

axs[1].plot(time, derivative)
# rolling_mean = pd.Series(derivative).rolling(window=window_size).mean()
# for value in time:
#     axs[1].plot(time[value], average_derivative)
fixed_line_length = average_derivative
axs[1].axhline(y=fixed_line_length, color='r', linestyle='--', label='Độ dài đường thẳng cố định')
# axs[1].plot(time, rolling_mean, label='Đường thẳng trung bình', linestyle='--')
# for value in ampl_pcg_data_filtered:
#     axs[1].plot(value, pcg_filtered[value], "r*")
axs[1].set_xlabel("Thời gian (s)")
axs[1].set_ylabel("Amplitude")
#axs[1].set_title("Đạo hàm 1")

axs[2].plot(time, derivative1)
# for value in ampl_pcg_data_filtered:
#     axs[1].plot(value, pcg_filtered[value], "r*")
axs[2].set_xlabel("Thời gian (s)")
axs[2].set_ylabel("Amplitude")
#axs[2].set_title("phát hiện tiếng ngáy")

plt.show()