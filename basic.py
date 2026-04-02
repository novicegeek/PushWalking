from contextlib import contextmanager
import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pywt
from scipy import signal
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import interp1d, pchip_interpolate
from scipy.ndimage import binary_dilation, gaussian_filter1d
import sympy as sp


class VoltConverter():
    def __init__(self, freq=5000):
        """
        Convert raw voltage data from sensors to mechanical quantities (forces/torques)
        """
        # Recalibrated by Thomas F Januar 2022 (given in V/5000 N)
        self.scale_factors = np.array([0.69793, 0.5698, 0.5536, 0.53573, 0.49673, 0.48505, 
                                       0.5954, 0.6032, 0.58867, 0.6513, 0.6032, 0.61773])
        self.scale_factors = 5000 / self.scale_factors
        self.r = 0.06375
        self.angles = (36, 60, 30)
        self.freq = freq

        # Extract angle values from input array and convert to radians
        a1_rad, a2_rad, a3_rad = np.deg2rad(self.angles[0]), np.deg2rad(self.angles[1]), np.deg2rad(self.angles[2])

        # Initialize 6x6 conversion matrix
        self.conversion_mat = np.zeros((6, 6))

        # Each column does a linear combination of the column vectors of the input data
        # to get an output metric
        # The first three columns are responsible for Fx, Fy and Fz
        self.conversion_mat[:, 0] = [np.cos(a1_rad) * np.sin(a3_rad),
                                     np.cos(a1_rad) * np.sin(a3_rad),
                                     -np.cos(a1_rad),
                                     np.cos(a1_rad) * np.sin(a3_rad),
                                     np.cos(a1_rad) * np.sin(a3_rad),
                                     -np.cos(a1_rad)]

        self.conversion_mat[:, 1] = [-np.cos(a1_rad) * np.cos(a3_rad),
                                     np.cos(a1_rad) * np.cos(a3_rad),
                                     0,
                                     -np.cos(a1_rad) * np.cos(a3_rad),
                                     np.cos(a1_rad) * np.cos(a3_rad),
                                     0]

        self.conversion_mat[:, 2] = np.sin(a1_rad) * np.ones(6)
        
        # The last three columns are responsible for Mx, My and Mz
        self.conversion_mat[:, 3] = self.r * np.array([np.sin(a1_rad) * np.cos(a2_rad),
                                                       -np.sin(a1_rad) * np.cos(a2_rad),
                                                       np.sin(a1_rad),
                                                       -np.sin(a1_rad) * np.cos(a2_rad),
                                                       np.sin(a1_rad) * np.cos(a2_rad),
                                                       np.sin(a1_rad)])
        
        self.conversion_mat[:, 4] = self.r * np.array([-np.sin(a1_rad) * np.sin(a2_rad),
                                                       -np.sin(a1_rad) * np.sin(a2_rad),
                                                       0,
                                                       np.sin(a1_rad) * np.sin(a2_rad),
                                                       np.sin(a1_rad) * np.sin(a2_rad),
                                                       0])

        self.conversion_mat[:, 5] = self.r * np.cos(a1_rad) * np.array([-1, 1, -1, 1, -1, 1])

    def volt_to_mechanics(self, volt_data: np.ndarray, spectrum_subtract, silence, reaction_force=True, **kwargs):
        """
        Convert raw voltage data from sensors to mechanical quantities (forces/torques)

        Parameters
        ----------
        volt_data : numpy.ndarray
            Raw voltage data from sensors, shape (n_samples, n_sensors)
        reaction_force: boolean
            `True` to return the reaction force (the pushing force to an external object)
              or `False` to return the action force (the force acting on the pad)

        Returns
        -------
        out_mechanics : numpy.ndarray
            Converted mechanical data, shape (n_samples, 6) (only use the first 6 channels)
            Typically represents [Fx, Fy, Fz, Mx, My, Mz]
        """
        # Ensure volt_data is 2D array
        volt_data = np.asarray(volt_data)[:, :6].copy()
        if not volt_data.any():
            return None
        
        # Filter each channel of the raw data to remove the noise
        volt_data_filtered = Filter.filter_combo(volt_data, self.freq, axis=0, **kwargs)

        # Use spectral subtraction to remove the walking noise from the signal
        if spectrum_subtract:
            volt_data_filtered = spectral_subtraction_same_trial(volt_data_filtered, self.freq, axis=0)

        # Silence the non-pushing period to suppress the noise
        if silence:
            volt_data_filtered = Mask.multi_impact_gating(volt_data_filtered, self.freq, axis=0)

        # Apply scale factors to raw voltage data (first 6 channels only)
        # Element-wise multiplication with broadcast
        scaled_volt = volt_data_filtered[:, :6] * self.scale_factors[:6]

        # Perform matrix multiplication: (n_samples x 6) @ (6 x 6) = (n_samples x 6)
        # Result contains transformed mechanical quantities
        out_mechanics = scaled_volt @ self.conversion_mat

        if reaction_force:
            return -out_mechanics
        else:
            return out_mechanics


def spectral_subtraction_same_trial(data, fs, axis):
    """
    Subtract the baseline spectrum from the whole signal.
    """
    def _spectral_subtraction_same_trial_id(data_1d):
        # 提取“指纹”段
        # 避开了推力的爆发期，又抓到了纯步行的特征
        noise_fingerprint = data_1d[round(fs*0.1) : round(fs*0.6)]
        
        # 如果前面不够长，就取推力结束后的段落补齐
        if len(noise_fingerprint) < fs * 0.3:
            noise_fingerprint = data_1d[len(data_1d)-round(fs*0.6) : len(data_1d)-round(fs*0.1)]
            
        # 4. 调用之前的谱减法函数
        cleaned_data = spectral_subtraction(data_1d, noise_fingerprint, fs=fs, axis=axis, alpha=2.0)
        
        return cleaned_data
    
    if data.ndim == 1:
        return _spectral_subtraction_same_trial_id(data)
    else:
        return np.apply_along_axis(_spectral_subtraction_same_trial_id, axis=axis, arr=data)
    

def spectral_subtraction(noisy_signal, noise_only_signal, fs, axis, alpha=2.0, beta=0.01):
    """
    使用谱减法去除背景步行干扰
    
    参数:
    - noisy_signal: 包含推力+步行的信号 (1D array)
    - noise_only_signal: 只有步行的参考信号 (1D array)
    - fs: 采样率
    - alpha: 减法过度因子 (Over-subtraction factor)。
             >1 会更干净但也可能损伤信号；通常取 1.0~3.0
    - beta: 谱下限 (Spectral floor)。
            防止减得太狠出现“负数”导致波形破碎，给基线留一点点极微弱的底噪。
    """
    def _spectral_subtraction_1d(noisy_signal_1d, noise_only_signal_1d):
        # 1. 计算 STFT (短时傅里叶变换)
        # nperseg 建议选 1024 (约 200ms)，能覆盖步行的特征周期
        nperseg = 1024
        f, t, Zxx_noisy = signal.stft(noisy_signal_1d, fs=fs, nperseg=nperseg)
        _, _, Zxx_noise = signal.stft(noise_only_signal_1d, fs=fs, nperseg=nperseg)
        
        # 2. 提取幅度和相位
        magnitude_noisy = np.abs(Zxx_noisy)
        phase_noisy = np.angle(Zxx_noisy)
        
        # 计算噪声信号的平均能量谱（均值背景）
        # 我们假设步行的特征在短时间内是稳定的
        magnitude_noise_mean = np.mean(np.abs(Zxx_noise), axis=1, keepdims=True)
        
        # 3. 执行谱减 (Power Subtraction)
        # 计算公式：Result_Mag = Noisy_Mag^2 - alpha * Noise_Mag^2
        # 我们用能量（幅度的平方）来计算
        mag_noisy_sq = magnitude_noisy**2
        mag_noise_sq = magnitude_noise_mean**2
        
        # 执行减法
        mag_result_sq = mag_noisy_sq - alpha * mag_noise_sq
        
        # 4. 半波整流与谱下限处理
        # 如果减成了负数，就设为 beta 比例的原始能量，防止产生诡异的“音乐噪声”
        mag_result_sq = np.where(mag_result_sq > beta * mag_noisy_sq, 
                                 mag_result_sq, 
                                 beta * mag_noisy_sq)
        
        mag_result = np.sqrt(mag_result_sq)
        
        # 5. 重构信号 (ISTFT)
        # 使用处理后的幅度 + 原始的相位
        Zxx_result = mag_result * np.exp(1j * phase_noisy)
        _, cleaned_signal = signal.istft(Zxx_result, fs=fs, nperseg=nperseg)
        
        # 保证长度一致
        return cleaned_signal[:len(noisy_signal_1d)]
    
    if noisy_signal.ndim == 1:
        return _spectral_subtraction_1d(noisy_signal, noise_only_signal)
    else:
        return np.apply_along_axis(_spectral_subtraction_1d, axis=axis, arr=noisy_signal, noise_only_signal_1d=noise_only_signal)
    

class Mask():
    def __init__():
        pass

    @classmethod
    def multi_impact_gating(cls, data, fs, axis, threshold_factor=0.1, min_dist_s=0.1):
        """
        支持多峰值检测的信号屏蔽门限函数
        
        参数:
        - data: 滤波后的力数据 (1D)
        - fs: 采样率
        - threshold_factor: 判定为信号开始/结束的阈值系数（相对于局部峰值）
        - min_dist_ms: 两个推力之间允许的最小间隔（防止把一个带锯齿的峰切成两半）
        """
        def _multi_impact_gating_1d(data_1d):
            # 1. 寻找所有显著的峰值（包括负值）
            # distance: 两次推力之间至少间隔的时间（s）
            # prominence: 峰值必须比周围高出一定程度，防止误触步行的波浪
            buffer = round(0.2 * fs)
            data_1d_no_buffer = data_1d[buffer : len(data_1d-buffer)]
            peaks, properties = signal.find_peaks(np.abs(data_1d_no_buffer), 
                                                  height=np.max(np.abs(data_1d_no_buffer))*0.7, # 只看高于全局最大值一定比例的峰
                                                  distance=int(min_dist_s * fs),
                                                  prominence=np.std(data_1d_no_buffer)*2)
            peaks += buffer
            
            if len(peaks) == 0:
                return np.zeros_like(data_1d)

            # 创建一个全局掩码（Mask），初始化为全 False
            total_mask = np.zeros(len(data_1d), dtype=bool)
            
            # 2. 遍历每一个检测到的峰，寻找各自的边界
            for p in peaks:
                p_val = data_1d[p]
                # 局部基线：取该峰值前/后200ms的均值中较接近峰值者
                if p_val >= 0:
                    local_base = max(np.mean(data_1d[max(0, round(p-fs*0.2)) : max(0, round(p-fs*0.1))]),
                                    np.mean(data_1d[min(round(p+fs*0.1), len(data_1d)-1) : min(round(p+fs*0.2), len(data_1d)-1)]))
                else:
                    local_base = min(np.mean(data_1d[max(0, round(p-fs*0.2)) : max(0, round(p-fs*0.1))]),
                                    np.mean(data_1d[min(round(p+fs*0.1), len(data_1d)-1) : min(round(p+fs*0.2), len(data_1d)-1)]))
                threshold = local_base + (p_val - local_base) * threshold_factor
                
                # 向左寻找起点
                start = p
                while start > 0 and (data_1d[start] - threshold) * sp.sign(p_val) > 0:
                    start -= 1
                
                # 向右寻找终点
                end = p
                while end < len(data_1d)-1 and (data_1d[end] - threshold) * sp.sign(p_val) > 0:
                    end += 1
                    
                # 将这个推力的区间加入掩码
                total_mask[start:end] = True
                
            # 3. 掩码优化：向外扩张半个最小峰间距，确保覆盖完整的起始/结束细节
            # 同时也把靠得很近的两个小峰“连通”起来
            total_mask = binary_dilation(total_mask, iterations=round(fs*min_dist_s/2))
            
            # 局部淡入淡出（简单实现：直接使用平滑后的 mask）
            smooth_mask = gaussian_filter1d(total_mask.astype(float), sigma=50)
            
            return data_1d * smooth_mask

        if data.ndim == 1:
            return _multi_impact_gating_1d(data)
        else:
            return np.apply_along_axis(_multi_impact_gating_1d, axis=axis, arr=data)

    @classmethod
    def apply_force_gate(cls, data, fs, axis, fade_len=250):
        """
        Zero all values except the pushing period, and apply fading at boundaries for smooth transition.

        :param fade_len: Length of fading zone.
        """
        def _apply_force_gate_1d(data_1d):
            start_idx, end_idx = cls.detect_impact_bounds(data_1d, fs, axis)

            gated_data = np.zeros_like(data_1d)
            
            # Copy pushing period data
            gated_data[start_idx:end_idx] = data_1d[start_idx:end_idx]
            
            # Fade in and out to prevent spetrum leak and sudden change of values
            if start_idx > fade_len:
                fade_in = np.linspace(0, 1, fade_len)
                gated_data[start_idx-fade_len:start_idx] = data_1d[start_idx-fade_len:start_idx] * fade_in
                
            if end_idx < len(data_1d) - fade_len:
                fade_out = np.linspace(1, 0, fade_len)
                gated_data[end_idx:end_idx+fade_len] = data_1d[end_idx:end_idx+fade_len] * fade_out
                
            return gated_data
        
        if data.ndim == 1:
            return _apply_force_gate_1d(data)
        else:
            return np.apply_along_axis(_apply_force_gate_1d, axis=axis, arr=data)

    @classmethod
    def detect_impact_bounds(cls, data, fs, axis, threshold_factor=0.1):
        """
        Automatically detect the indices of start and end of impulse signal.
        """
        def _detect_impact_bounds_1d(data_1d):
            """
            Apply to each 1d array.
            """
            # Slightly smooth the signal to avoid peak artifacts
            smooth_data = gaussian_filter1d(data_1d, sigma=10) 
            
            # Find the peak of the signal (including negative peak)
            # Exclude buffer period in the beginning and end to avoid boundary effects of filter
            buffer = round(0.1 * fs)
            peak_idx = np.argmax(np.abs(smooth_data[buffer : len(smooth_data) - buffer])) + buffer
            peak_val = smooth_data[peak_idx]
            
            # Set dynamic threshold: Criterium relative to peak value
            # Take a certain proportion between local baseline and peak to account for the fluctuation
            local_baseline = np.mean(smooth_data[max(0, peak_idx-round(0.5*fs)) : max(0, peak_idx-round(0.2*fs))])
            threshold = local_baseline + (peak_val - local_baseline) * threshold_factor
            
            # Find the start index to the left
            start_idx = peak_idx
            while start_idx > 0:
                if (smooth_data[start_idx] - threshold) * sp.sign(peak_val) < 0:
                    break
                start_idx -= 1
                
            # Find the end index to the right
            end_idx = peak_idx
            while end_idx < len(smooth_data) - 1:
                if (smooth_data[end_idx] - threshold) * sp.sign(peak_val) < 0:
                    break
                end_idx += 1
                
            return start_idx, end_idx
        
        if data.ndim == 1:
            return _detect_impact_bounds_1d(data)
        else:
            indices = np.apply_along_axis(_detect_impact_bounds_1d, axis=axis, arr=data)
            start_indices = get_slice(indices, axis, 0)
            end_indices = get_slice(indices, axis, 1)
            return start_indices, end_indices

    @classmethod
    def variance_based_silence(cls, data, axis, window_size=500, threshold_ratio=0.5):
        """
        Silence the non-pushing period based on rolling variance.
        """
        # Calculate rolling variance
        rolling_std = np.stack([np.std(np.take(data, np.arange(i, i+window_size), axis=axis), axis=axis) 
                                for i in range(data.shape[axis] - window_size + 1)], axis=axis)

        # Pad the array to match the length of input data
        rolling_std = np.apply_along_axis(np.pad, axis=axis, arr=rolling_std, pad_width=(0, window_size - 1), mode='edge')
        
        # Keep periods that have std greater than a specific value between min std and max std
        threshold = np.min(rolling_std, axis=axis, keepdims=True) + \
            (np.max(rolling_std, axis=axis, keepdims=True) - np.min(rolling_std, axis=axis, keepdims=True)) * threshold_ratio
        mask = rolling_std > threshold
        
        # Smooth the mask by dilation
        mask = binary_dilation(mask, iterations=round(window_size/2), axes=axis)  # Expand to cover the whole wave

        # Expand dimension for multiplication
        # mask = np.expand_dims(mask, list(range(data.ndim)).pop(axis))
        return data * mask
    

class Filter():
    def __init__():
        pass

    @classmethod
    def filter_combo(cls, data, freq, axis, notchfilt=True, medfilt=False, movemean=False, 
                     waveletdenoise=False, sgfilt=False, 
                     butterfilt=True, butter_cutoff=15, butter_type='low'):
        """
        Reduce noises in the data by using combined filters.

        :param array-like data: The data to be filtered.
        :param freq: Sampling frequency of the data.
        :param axis: The axis to apply filter along.
        """
        # Baseline removal / detrending
        # Some filters are sensitive to offset, therefore it needs to be removed prior to filtering
        data = remove_offset(data, freq, axis)

        # Notch filter at 25 Hz: Remove the highest-intensity noise of fixed frequency 
        # (probably coming from electric circuits)
        if notchfilt:
            for f in [25, 50]:
                b_notch, a_notch = signal.iirnotch(f, 30, freq)
                data = signal.filtfilt(b_notch, a_notch, data, axis=axis)
        
        # Median filter: Remove the spikes caused by AD conversion or BNC connectors
        if medfilt:
            data = cls.median_filt(data, 5, axis)

        # Moving average: Flatten the undulations
        if movemean:
            for _ in range(4):  # Calculate 4 pass moving average to approximate a Gaussian filter kernel
                data = cls.moving_average_convolve(data, 11, axis)

        # Wavelet denoising: Remove complex noise in data with acute signals
        if waveletdenoise:
            data = cls.wavelet_denoising(data, 'sym8', 7, axis)

        # Savitzky-Golay filter: Remove wide-band white noise
        if sgfilt:
            data = signal.savgol_filter(data, window_length=251, polyorder=3, axis=axis)

        # Butterworth filter
        if butterfilt:
            data = cls.butter_filt(data, butter_cutoff, freq, btype=butter_type, axis=axis)

        return data

    @classmethod
    def median_filt(cls, data, kernel_size, axis):
        """
        Apply median filter to individual channels of the data.
        """
        if data.ndim == 1:
            return signal.medfilt(data, kernel_size)
        else:
            return np.apply_along_axis(signal.medfilt, axis=axis, arr=data, kernel_size=kernel_size)

    @classmethod
    def moving_average_convolve(cls, data, window_size, axis):
        """
        Smooth data by moving average.
        """
        # Create filter kernel: length is window_size, each component is 1/window_size
        kernel = np.ones(window_size) / window_size

        # Use convolution function
        if data.ndim == 1:
            return np.convolve(data, kernel, mode='same')
        else:
            return np.apply_along_axis(np.convolve, axis=axis, arr=data, v=kernel, mode='same')

    @classmethod
    def wavelet_denoising(cls, data, wavelet, level, axis):
        """
        Apply wavelet denoising to data.
        """
        # Core denoising function for 1D data (internal use only)
        def _wavelet_denoising_1d(data_1d):
            # Decomposing
            coeffs = pywt.wavedec(data_1d, wavelet, level=level)
            
            # Calculate threshold and process detail component
            # Use Universal Threshold (Sigma * sqrt(2*log(n)))
            sigma = (np.median(np.abs(coeffs[-1])) / 0.6745)
            threshold = sigma * np.sqrt(2 * np.log(len(data_1d)))

            # Apply soft threshold to detail component
            new_coeffs = [coeffs[0]]  # Retain approximation component
            for i in range(1, len(coeffs)):
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

            # Reconstruct signal
            cleaned_data_1d = pywt.waverec(new_coeffs, wavelet)
            return cleaned_data_1d[:len(data_1d)]  # Keep the original length
        
        if data.ndim == 1:
            return _wavelet_denoising_1d(data)
        else:
            return np.apply_along_axis(_wavelet_denoising_1d, axis=axis, arr=data)

    @classmethod
    def butter_filt(cls, data, cutoff_freq, sampling_freq, btype='low', order=2, output='sos', axis=-1, padlen=0.2):
        """
        Apply Butterworth filter to data.

        :param padlen: Length of padding for filter in second.
        """
        filt = signal.butter(order, Wn=cutoff_freq, btype=btype, output=output, fs=sampling_freq)
        padlen = round(padlen * sampling_freq)

        if isinstance(data, dict):
            data_filtered = {field: signal.sosfiltfilt(filt, field_data, axis=axis, padlen=padlen) for field, field_data in data.items()}
        else:
            data_filtered = signal.sosfiltfilt(filt, data, axis=axis, padlen=padlen)
        return data_filtered
    

def extract_push_range(data_1d, fs, f_threshold=1, window_size=10, max_dist_from_peak=0.25):
    """
    Extract the start, peak and end of a push, assuming that the data were already filtered and gated, 
    and only one push exists.

    :param data_1d: Must be a 1d array
    :param window_size: The number of consecutive frames used to determine a start/end frame.
    :param max_dist_from_peak: The maximum distance of the start/end frame from the peak frame, in second
    :return: (start_idx, peak_idx, end_idx)
    """
    buffer = 0
    max_dist_frames = round(max_dist_from_peak * fs)
    peak_idx = np.argmax(np.abs(data_1d))
    start_idx = buffer
    end_idx = len(data_1d) - 1 - buffer
    while start_idx < peak_idx:
        if (min(np.abs(data_1d[start_idx:start_idx+window_size])) > f_threshold and 
            (peak_idx - start_idx <= max_dist_frames or start_idx == buffer)):
            break
        start_idx += 1
    while end_idx > peak_idx:
        if (min(np.abs(data_1d[end_idx-window_size+1:end_idx+1])) > f_threshold and
            (end_idx - peak_idx <= max_dist_frames or end_idx == len(data_1d) - 1 - buffer)):
            break
        end_idx -= 1
    return [start_idx, peak_idx, end_idx]


def divide_by_threshold(numerator_arr, denominator_arr, threshold):
    """
    Perform division and return the quotient when the absolute value of denominator is larger than `threshold`.
    Otherwise, the result is set to 0.
    """
    results = np.where(np.abs(denominator_arr) > threshold, numerator_arr / denominator_arr, 0.0)
    return results


def fix_cop_outliers(data, min_bound, max_bound, axis):
    """
    剔除并插值填充 COP 异常点
    """
    def _fix_cop_outliers_1d(data_1d):
        cleaned_cop = data_1d.copy()
        
        # 1. 识别异常点：超出物理范围或数值突变
        # 这里以物理边界为例
        outliers = (data_1d < min_bound) | (data_1d > max_bound)
        
        # 2. 将异常点设为 NaN (空值)
        cleaned_cop[outliers] = np.nan
        
        # 3. 寻找非空（正常）点的索引和数值
        idx = np.arange(len(cleaned_cop))
        is_valid = ~np.isnan(cleaned_cop)
        
        # 4. 如果全都是异常点，直接返回 0
        if not np.any(is_valid):
            return np.zeros_like(cleaned_cop)
        
        # 5. 执行插值填充 (使用线性 'linear' 或 立方样条 'cubic')
        # cubic 更加平滑，但如果连续异常点太多可能会产生过冲
        f = interp1d(idx[is_valid], cleaned_cop[is_valid], 
                    kind='linear', fill_value="extrapolate")
        
        return f(idx)
    
    if data.ndim == 1:
        return _fix_cop_outliers_1d(data)
    else:
        return np.apply_along_axis(_fix_cop_outliers_1d, axis=axis, arr=data)


def remove_offset(data, freq, axis, source=None) -> dict | np.ndarray:
    """
    Remove the offset of data by subtracting the initial average value, using the first 0.1 to 0.3 second data.

    :param data: The input data. Should be dict or numpy.ndarray.
    :param source: The source of input data. Can be `'markers'`, `'force plate'`, `'pad'` or None.
    :param freq: The sampling frequency of the data.
    :param axis: The axis to remove offset along.
    :return: Dict or numpy.ndarray of the data after removing offset.
    """
    i_start_frame = round(0.1 * freq)  # Start from the first 0.1 second
    i_end_frame = round(0.3 * freq)  # End at the first 0.3 second
    if source == 'markers':
        print("Marker data doesn't need to remove offset. The input data is returned.")
        data_offset = data
    elif isinstance(data, np.ndarray):
        offset = np.mean(np.take(data, np.arange(i_start_frame, i_end_frame), axis=axis), axis=axis, keepdims=True)
        data_offset = data - offset
    elif isinstance(data, dict):
        data_offset = {}
        for field, field_data in data.items():
            if field == 'center_of_pressure':
                data_offset[field] = field_data  # Center of pressure data doesn't need to remove offset
            else:
                offset = np.mean(np.take(field_data, np.arange(i_start_frame, i_end_frame), axis=axis), axis=axis, keepdims=True)
                data_offset[field] = field_data - offset
    else:
        print(f"Invalid object type {type(data)} to remove offset")
        data_offset = None
    return data_offset


def rotate_coordinate_system(data_array, rotate_axis, rotate_angle, axis_dim):
    """
    Rotate coordinate system of input data array by `rotate_angle` around `rotate_axis` in `axis_dim`.
    Any metrics expressed in 3D components (e.g., forces, torques, marker positions) can be rotated.

    Parameters
    ----------
    data_array : numpy.ndarray
        Input data array.
    rotate_axis : str
        Axis to rotate around ('x', 'y', or 'z').
    rotate_angle : float
        Angle to rotate in degrees. Note: The coordinate system will be rotated by this angle 
        according to the right-hand rule, instead of the data vectors themselves.
    axis_dim : int
        The dimension index in `data_array` that corresponds to the 3D components to be rotated.

    Returns
    -------
    rotated_data : numpy.ndarray
        Data array with the rotated data of the same number and order of dimensions as `data_array`.
    """
    # Convert angle to radians
    angle_rad = np.deg2rad(rotate_angle)

    # Define rotation matrix
    match rotate_axis:
        case 'x':
            rot_matrix = np.array(
                [[1, 0, 0],
                [0, sp.cos(angle_rad), sp.sin(angle_rad)],
                [0, -sp.sin(angle_rad), sp.cos(angle_rad)]]
            )
        case 'y':
            rot_matrix = np.array(
                [[sp.cos(angle_rad), 0, -sp.sin(angle_rad)],
                [0, 1, 0],
                [sp.sin(angle_rad), 0, sp.cos(angle_rad)]]
            )
        case 'z':
            rot_matrix = np.array(
                [[sp.cos(angle_rad), sp.sin(angle_rad), 0],
                [-sp.sin(angle_rad), sp.cos(angle_rad), 0],
                [0, 0, 1]]
            )
    rot_matrix = rot_matrix.astype(float)  # Convert symbolic matrix to numeric matrix for efficient computation

    # Swap the dimension with 3D components with the last dimension for matrix multiplication
    data_array_swapped = np.swapaxes(data_array, axis_dim, -1)
    data_array_rotated_swapped = data_array_swapped @ rot_matrix.T
    # Sway again to restore the dimension order
    data_array_rotated = np.swapaxes(data_array_rotated_swapped, axis_dim, -1)
    
    return data_array_rotated


def transform_pad_to_global(pad_data, marker_data, mocap_metadata):
    """
    Transform the boxing pad data from the local (pad) coordinate system to global (lab) coordinate system.
    The input `mocap_data` must be represented in global (lab) coordinate system.

    The (conceptual) axis orientations of the pad coordinate system are as follows (Actual orientation 
    after construction may have trivial deviation due to marker attachment and motion capture errors):

    +X - Parallel to the (ideal) short axis of the pad surface, 
    pointing from bottom to top (roughly from the midpoint of bottom markers to the midpoint of top markers)

    +Y - Parallel to the (ideal) long axis of the pad surface, 
    pointing from left to right (roughly from the midpoint of left markers to the midpoint of right markers)

    +Z - Perpendicular to the pad surface, pointing from the pushing side to the handle side
    """
    if pad_data is None:
        return None
    n_frames = pad_data['force'].shape[1]

    (topleft_in_global, 
    bottomleft_in_global, 
    topright_in_global, 
    bottomright_in_global, 
    center_in_global) = get_pad_center(marker_data, mocap_metadata)

    # Build +Y axis
    midleft_in_global = (topleft_in_global + bottomleft_in_global) / 2
    midright_in_global = (topright_in_global + bottomright_in_global) / 2
    y_local = midright_in_global - midleft_in_global  # Y-axis of local coordinate system expressed in global coordinate system
    y_local_normalized = y_local / np.linalg.norm(y_local, axis=0, keepdims=True)

    # Build +Z axis
    helper_vec = topright_in_global - bottomright_in_global  # Build helper vector
    z_local = np.cross(helper_vec, y_local_normalized, axis=0)
    z_local_normalized = z_local / np.linalg.norm(z_local, axis=0, keepdims=True)

    # Built +X axis
    x_local = np.cross(y_local_normalized, z_local_normalized, axis=0)             # The x_local should already be unit vectors in principle, but due to 
    x_local_normalized = x_local / np.linalg.norm(x_local, axis=0, keepdims=True)  # numeric errors the normalization is done again to secure this

    # Build transformation matrix
    conversion_mats_translation_orig = np.stack([x_local_normalized, y_local_normalized, z_local_normalized, center_in_global], axis=2)
    conversion_mats_translation = quick_interpolate(conversion_mats_translation_orig, pad_data['force'])
    conversion_mats = conversion_mats_translation[:, :, :3]
    converted_data = {
        "force": np.einsum('ikj,jk->ik', conversion_mats, pad_data['force']),
        "center_of_pressure": np.einsum('ikj,jk->ik', conversion_mats_translation, np.row_stack([pad_data['center_of_pressure'], np.ones(n_frames)])),
        "moment": np.einsum('ikj,jk->ik', conversion_mats, pad_data['moment'])
    }
    
    return converted_data


def get_pad_center(marker_data, mocap_metadata):
    """
    Get the coordinate data of the geometric center of the boxing pad.
    """
    # Extract the indices for the four corner markers on the pad in the marker data
    labels_list = mocap_metadata['marker_labels']
    topleft_idx = labels_list.index('Pad_TopLeft')
    bottomleft_idx = labels_list.index('Pad_BottomLeft')
    topright_idx = labels_list.index('Pad_TopRight')
    bottomright_idx = labels_list.index('Pad_BottomRight')

    # Extract the markers coordinate data in the global(lab) coordinate system
    topleft = marker_data[:, topleft_idx, :]
    bottomleft = marker_data[:, bottomleft_idx, :]
    topright = marker_data[:, topright_idx, :]
    bottomright = marker_data[:, bottomright_idx, :]

    # Calculate boxing pad center (as the origin of local coordinate system) for COP transformation
    center = np.mean(np.stack([topleft, bottomleft, topright, bottomright], axis=2), 
                     axis=2)
    
    return topleft, bottomleft, topright, bottomright, center


def get_nearest_index(ind: int | list[int], fs_orig, fs_target):
    """
    Get the index of frame in target array nearest in time to that in reference array.
    """
    if not isinstance(ind, list | tuple):
        ind = [ind]
    return [round((i+1)/fs_orig*fs_target)-1 if not math.isnan(i) else math.nan for i in ind]


def get_slice(arr, axis, index):
    """
    Get the slice of `arr` at position `index` along `axis`, keeping the other dimensions complete.
    """
    # Create an array filled with colons, of the same length of the number of dimensions of the input `arr`
    indices = [slice(None)] * arr.ndim
    
    # Replace specified axis to target index
    indices[axis] = index
    
    # Convert the list to tuple for indexing
    return arr[tuple(indices)]


def get_vector_norm(data, axis=0):
    """
    Get the 2-norm of a vector (vector sum from 3 axis)
    """
    if isinstance(data, np.ndarray):
        return np.linalg.norm(data, axis=axis)
    elif isinstance(data, dict):
        return {field: np.linalg.norm(field_data, axis=axis) for field, field_data in data.items()}
    elif math.isnan(data):
        return math.nan
    else:
        return None


def integrate_interval(data, start_idx, end_idx, fs, axis=-1, rule='trapezoid'):
    if any([math.isnan(i) for i in (start_idx, end_idx)]):
        return math.nan
    time_array = np.arange(start_idx, end_idx + 1) / fs
    if rule == 'trapezoid':
        return trapezoid(np.take(data, np.arange(start_idx, end_idx + 1), axis=axis), time_array, axis=axis)
    if rule == 'simpson':
        return simpson(np.take(data, np.arange(start_idx, end_idx + 1), axis=axis), time_array, axis=axis)


def quick_interpolate(orig_array, template_array, kind=None, time_axis=1):
    """
    Interpolate the `orig_array` to be as long as `template_array` by method specified by `kind`.
    """
    orig_len = orig_array.shape[time_axis] if orig_array.ndim > 1 else len(orig_array)
    new_len = template_array.shape[time_axis] if template_array.ndim > 1 else len(template_array)
    orig_indices = np.arange(1, orig_len + 1)
    new_indices = np.linspace(orig_len/new_len, orig_len, new_len)
    if not kind:  # Use pchip by default
        interp_array = pchip_interpolate(orig_indices, orig_array, new_indices, axis=time_axis)
    else:
        f = interp1d(orig_indices, orig_array, kind, axis=time_axis)
        interp_array = f(new_indices)
    return interp_array


def find_response_moment(moment_array, baseline_start_idx, anchor_step_idx, end_idx):
    """
    Find the response moment by finding the highest peak in the step following anchor step.
    Note: The function finds maximum peak. Therefore, the input should be taken negative
    values if minimum peak is to be found.

    :param baseline_start_idx: The frame index of the start of the stride preceding anchor step.
    :param anchor_step_idx: The frame index of anchor step.
    :param end_idx: The frame index of the end of the step following anchor step.
    :return: Tuple of (index, value).
    """
    # Exclude 0.1 * baseline stride in the beginning to avoid capturing 
    # small fluctuation (specifically for ankle)
    start_idx = round(anchor_step_idx + 0.1 * (anchor_step_idx - baseline_start_idx))
    peaks = signal.find_peaks(moment_array[start_idx:end_idx+1])[0]
    peaks += start_idx

    # Get the highest peak and index
    # Results are in descending order of values
    top_with_idx = heapq.nlargest(1, zip(peaks, moment_array[peaks]), key=lambda x: x[1])

    # Return the peak
    if top_with_idx:
        return top_with_idx[0]
    else:
        return (math.nan, math.nan)
    

@contextmanager
def temp_chdir(new_dir):
    """
    Context manager to temporarily change the current working directory

    Parameters
    ----------
    new_dir : str
        The directory to temporarily change to
    """
    original_dir = os.getcwd()
    os.chdir(new_dir)
    try:
        yield
    finally:
        os.chdir(original_dir)


@contextmanager
def suppress_cpp_stdout(stdout_path='', remove_old=True):
    """Suppress GUI IO and redirect it to other output if specified"""
    # Open devnull if no IO is specified
    if not os.path.isfile:
        new_stdout_fd = os.open(os.devnull, os.O_WRONLY)
    else:
        import io
        flags = os.O_CREAT | os.O_WRONLY | os.O_TRUNC
        new_stdout_fd = os.open(stdout_path, flags)
    # Backup the original stdout file descriptor (normally 1)
    old_stdout_fd = os.dup(1)
    try:
        # Redirect stdout (1) to devnull
        os.dup2(new_stdout_fd, 1)
        yield
    finally:
        # Restore the original stdout
        os.dup2(old_stdout_fd, 1)
        os.close(new_stdout_fd)
        os.close(old_stdout_fd)