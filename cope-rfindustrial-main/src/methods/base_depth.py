from glob import glob

import numpy as np
from scipy import signal

from src.data.utils import load_sample_file
from src.data.utils import remove_saturation as remove_saturation_method
from src.methods.utils import get_input_signal


COUPLES_DISTANCE = 9.4
MIN_DEPTH = 300


class BaseDepthEstimationMethod:
    def predict_folder(self,
                       ecometry_trials,
                       estimated_speed=None,
                       use_correlation=False,
                       remove_saturation=True):

        used_signal = 0
        df_params = None
        output_signal_mean = np.array([])

        for ecometry_name in ecometry_trials:
            # Load signal.
            df_params, output_signal_original = load_sample_file(ecometry_name)

            if len(output_signal_mean) < len(output_signal_original):
                result = output_signal_original.copy()
                result[:len(output_signal_mean)] += output_signal_mean
            else:
                result = output_signal_mean.copy()
                result[:len(output_signal_original)] += output_signal_original
            output_signal_mean = np.array(result.copy())
            used_signal += 1

        print(f'Using {used_signal} of {len(ecometry_trials)} provided signals.')
        output_signal_original = np.array(output_signal_mean) / used_signal

        if remove_saturation:
            signal_len = len(output_signal_original)
            output_signal_original = remove_saturation_method(output_signal_original)
            removed_samples = signal_len - len(output_signal_original)
        else:
            removed_samples = 0

        estimated_depth = self.predict(
            output_signal_original,
            df_params,
            estimated_speed=estimated_speed,
            removed_samples=removed_samples,
            use_correlation=use_correlation)

        return estimated_depth

    def find_peaks(self,
                   final_output_signal,
                   peak_min_height,
                   peak_min_distance,
                   output_signal_sampling_freq,
                   removed_samples,
                   estimated_speed):
        # Compute peaks.
        peaks_ix, peaks = signal.find_peaks(final_output_signal,
                                            height=peak_min_height,
                                            distance=peak_min_distance)
        peaks = peaks['peak_heights']
        peaks_ix = (peaks_ix + removed_samples) / output_signal_sampling_freq

        # Remove peaks which are smaller than MIN_DEPTH.
        final_peaks_ix = []
        final_peaks = []
        for peak_ix, peak in zip(peaks_ix, peaks):
            if (peak_ix * estimated_speed) / 2 >= MIN_DEPTH:
                final_peaks_ix.append(peak_ix)
                final_peaks.append(peak)

        return final_peaks_ix, final_peaks


class InitialEstimation(BaseDepthEstimationMethod):
    def __init__(self,
                 f1=5,
                 f2=50,
                 fir_filter_order=400,
                 low_pass_filter_order=10,
                 low_pass_cutoff=50,
                 peak_min_height=0.5,
                 peak_min_distance=500):
        self.f1 = f1
        self.f2 = f2
        self.fir_filter_order = fir_filter_order
        self.low_pass_filter_order = low_pass_filter_order
        self.low_pass_cutoff = low_pass_cutoff
        self.peak_min_height = peak_min_height
        self.peak_min_distance = peak_min_distance

    def predict(self,
                output_signal_values,
                df_ecometry_params,
                estimated_speed,
                removed_samples=0,
                use_correlation=False):
        if use_correlation:
            input_signal_values, _ = get_input_signal(df_ecometry_params.Frec1,
                                                      df_ecometry_params.Down,
                                                      df_ecometry_params.IncF,
                                                      df_ecometry_params.Nciclos)
            output_signal = signal.correlate(output_signal_values, input_signal_values)

            # Remove input signal from correlation, plus an arbitrary number of samples.
            output_signal = output_signal[len(input_signal_values) + 10:]
        else:
            output_signal = output_signal_values

        # Normalize.
        output_signal_norm = output_signal / np.max(np.abs(output_signal))

        # Compute signal frequency.
        output_signal_sampling_freq = 930000 / df_ecometry_params.Down

        # Filter signal.
        fir_filter = signal.firwin(self.fir_filter_order + 1, [self.f1, self.f2], pass_zero=False,
                                   fs=output_signal_sampling_freq)
        output_signal_filtered = signal.lfilter(fir_filter, 1, output_signal_norm)

        # Get absolute value.
        output_signal_abs = np.abs(output_signal_filtered)

        # Apply a low-pass filter.
        b, a = signal.butter(self.low_pass_filter_order,
                             self.low_pass_cutoff,
                             btype='low',
                             analog=False,
                             fs=output_signal_sampling_freq)
        output_signal_filtered_low_pass = signal.filtfilt(b, a, output_signal_abs)

        peaks_ix, peaks = self.find_peaks(
            output_signal_filtered_low_pass,
            self.peak_min_height,
            self.peak_min_distance,
            output_signal_sampling_freq,
            removed_samples,
            estimated_speed)

        if len(peaks_ix) > 0:
            max_ix_value = peaks_ix[np.argmax(peaks)]
            estimated_depth = max_ix_value * estimated_speed / 2
        else:
            estimated_depth = None

        return estimated_depth


class InitialEstimationHilbert(BaseDepthEstimationMethod):
    def __init__(self,
                 f1=5,
                 f2=50,
                 order=100,
                 peak_min_height=0.5,
                 peak_min_distance=500):
        self.f1 = f1
        self.f2 = f2
        self.filter_order = order
        self.peak_min_height = peak_min_height
        self.peak_min_distance = peak_min_distance

    def predict(self,
                output_signal_values,
                df_ecometry_params,
                estimated_speed,
                removed_samples=0,
                use_correlation=False):
        if use_correlation:
            input_signal_values, _ = get_input_signal(df_ecometry_params.Frec1,
                                                      df_ecometry_params.Down,
                                                      df_ecometry_params.IncF,
                                                      df_ecometry_params.Nciclos)
            output_signal = signal.correlate(output_signal_values, input_signal_values)

            # Remove input signal from correlation, plus an arbitrary number of samples.
            output_signal = output_signal[len(input_signal_values) + 10:]
        else:
            output_signal = output_signal_values

        # Normalize.
        output_signal_norm = output_signal / np.max(np.abs(output_signal))

        # Compute signal frequency.
        output_signal_sampling_freq = 930000 / df_ecometry_params.Down

        # Band-pass filter
        fir_filter = signal.firwin(self.filter_order + 1, [self.f1, self.f2], pass_zero=False,
                                   fs=output_signal_sampling_freq)
        output_signal_filtered = signal.lfilter(fir_filter, 1, output_signal_norm)

        # Filter signal.
        filtered_signal_envelope = signal.hilbert(output_signal_filtered)
        filtered_signal_envelope = np.abs(filtered_signal_envelope)

        peaks_ix, peaks = self.find_peaks(
            filtered_signal_envelope,
            self.peak_min_height,
            self.peak_min_distance,
            output_signal_sampling_freq,
            removed_samples,
            estimated_speed)

        if len(peaks_ix) > 0:
            max_ix_value = peaks_ix[np.argmax(peaks)]
            estimated_depth = max_ix_value * estimated_speed / 2
        else:
            estimated_depth = None

        return estimated_depth


class RFIndustrialEstimation(BaseDepthEstimationMethod):
    def __init__(self,
                 peak_min_height=0.5,
                 peak_min_distance=500):
        self.peak_min_height = peak_min_height
        self.peak_min_distance = peak_min_distance

    def predict(self,
                ecometry_path,
                estimated_speed):
        df_params = None
        output_signal_mean = np.array([])

        ecometry_trials = glob(ecometry_path + '/*.json')

        for ecometry_name in ecometry_trials:
            # Load signal.
            df_params, output_signal_original = load_sample_file(ecometry_name)

            if len(output_signal_mean) < len(output_signal_original):
                result = output_signal_original.copy()
                result[:len(output_signal_mean)] += output_signal_mean
            else:
                result = output_signal_mean.copy()
                result[:len(output_signal_original)] += output_signal_original
            output_signal_mean = np.array(result.copy())

        input_signal_values, _ = get_input_signal(df_params.Frec1,
                                                  df_params.Down,
                                                  df_params.IncF,
                                                  df_params.Nciclos)
        output_signal = signal.correlate(output_signal_mean, input_signal_values)

        # Compute signal frequency.
        output_signal_sampling_freq = 930000 / df_params.Down

        # Remove input signal from correlation, plus an arbitrary number of samples.
        output_signal = output_signal[len(input_signal_values) + 10:]

        signal_len = len(output_signal)
        output_signal = remove_saturation_method(output_signal)
        removed_samples = signal_len - len(output_signal)

        peaks_ix, peaks = self.find_peaks(
            output_signal,
            self.peak_min_height,
            self.peak_min_distance,
            output_signal_sampling_freq,
            removed_samples,
            estimated_speed)

        if len(peaks_ix) > 0:
            max_ix_value = peaks_ix[np.argmax(peaks)]
            estimated_depth = max_ix_value * estimated_speed / 2
        else:
            estimated_depth = None

        return estimated_depth
