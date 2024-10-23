import numpy as np
from scipy import signal
from scipy.fft import fft, ifft

from src.data.utils import load_sample_file
from src.methods.utils import get_input_signal
from src.data.utils import remove_saturation as remove_saturation_method


COUPLES_DISTANCE = 9.4
MIN_DEPTH = 300


class InitialEstimation:
    def __init__(self,
                 remove_saturation=True,
                 f1=120,
                 f2=250,
                 fir_filter_order=700,
                 low_pass_filter_order=2,
                 low_pass_cutoff=70,
                 peak_min_height=0.1,
                 peak_min_distance=250):
        self.remove_saturation = remove_saturation
        self.f1 = f1
        self.f2 = f2
        self.fir_filter_order = fir_filter_order
        self.low_pass_filter_order = low_pass_filter_order
        self.low_pass_cutoff = low_pass_cutoff
        self.peak_min_height = peak_min_height
        self.peak_min_distance = peak_min_distance

    def predict(self,
                ecometry_name,
                compute_speed=True,
                estimated_speed=None,
                use_correlation=False):
        # Load signal.
        df, output_signal_original = load_sample_file(ecometry_name)
        if self.remove_saturation:
            output_signal_original = remove_saturation_method(output_signal_original)

        if use_correlation:
            input_signal_values, _ = get_input_signal(df.Frec1,
                                                      df.Down,
                                                      df.IncF,
                                                      df.Nciclos)
            output_signal = signal.correlate(output_signal_original, input_signal_values)

            # Remove input signal from correlation, plus an arbitrary number of samples.
            output_signal = output_signal[len(input_signal_values) + 10:]
        else:
            output_signal = output_signal_original

        # Normalize.
        output_signal_norm = output_signal / np.max(np.abs(output_signal))

        # Compute signal frequency.
        output_signal_sampling_freq = 930000 / df.Down

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

        # Compute peaks.
        peaks_ix, peaks = signal.find_peaks(output_signal_filtered_low_pass,
                                            height=self.peak_min_height,
                                            distance=self.peak_min_distance)
        peaks = peaks['peak_heights']
        peaks_ix = peaks_ix / output_signal_sampling_freq

        if len(peaks_ix) > 1:
            peaks_average_diff = np.average(np.diff(peaks_ix))
            estimated_speed = COUPLES_DISTANCE / peaks_average_diff
        else:
            estimated_speed = None

        return estimated_speed


    def predict_folder(self,
                       ecometry_trials,
                       compute_speed=True,
                       estimated_speed=None,
                       use_correlation=False):

        used_signal = 0

        for ecometry_name in ecometry_trials:
            # Load signal.
            df, output_signal_original = load_sample_file(ecometry_name,
                                                          remove_saturation=self.remove_saturation)

            if used_signal == 0:
                output_signal_mean = np.array(output_signal_original)
                used_signal += 1
            else:
                if (len(output_signal_mean) <= len(output_signal_original)):
                    output_signal_original = output_signal_original[:len(output_signal_mean)]
                    used_signal += 1
                    output_signal_mean = output_signal_mean + np.array(output_signal_original)

        print(f'Using {used_signal} of {len(ecometry_trials)} provided signals.')
        output_signal_original = np.array(output_signal_mean) / used_signal

        if use_correlation:
            input_signal_values, _ = get_input_signal(df.Frec1,
                                                      df.Down,
                                                      df.IncF,
                                                      df.Nciclos)
            output_signal = signal.correlate(output_signal_original, input_signal_values)

            # Remove input signal from correlation, plus an arbitrary number of samples.
            output_signal = output_signal[len(input_signal_values) + 10:]
        else:
            output_signal = output_signal_original

        # Normalize.
        output_signal_norm = output_signal / np.max(np.abs(output_signal))

        # Compute signal frequency.
        output_signal_sampling_freq = 930000 / df.Down

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

        # Compute peaks.
        peaks_ix, peaks = signal.find_peaks(output_signal_filtered_low_pass,
                                            height=self.peak_min_height,
                                            distance=self.peak_min_distance)
        peaks = peaks['peak_heights']
        peaks_ix = peaks_ix / output_signal_sampling_freq

        if compute_speed:
            if len(peaks_ix) > 1:
                peaks_average_diff = np.average(np.diff(peaks_ix))
                estimated_speed = COUPLES_DISTANCE / peaks_average_diff
            else:
                estimated_speed = None

            return estimated_speed
        else:
            if self.remove_saturation:
                _, output_signal_complete = load_sample_file(ecometry_name,
                                                             False)
                removed_samples = len(output_signal_complete) - len(output_signal_original)
                removed_samples = removed_samples / output_signal_sampling_freq
            else:
                removed_samples = 0

            final_peaks_ix = []
            final_peaks = []
            for peak_ix, peak in zip(peaks_ix, peaks):
                if (peak_ix + removed_samples) * estimated_speed / 2 >= MIN_DEPTH:
                    final_peaks_ix.append(peak_ix)
                    final_peaks.append(peak)
            peaks_ix = final_peaks_ix
            peaks = final_peaks

            if len(peaks_ix) > 0:
                max_ix_value = peaks_ix[np.argmax(peaks)]

                # Add removed samples.
                max_ix_value += removed_samples
                estimated_depth = max_ix_value * estimated_speed / 2
            else:
                estimated_depth = None

            return estimated_depth


class InitialEstimationHilbert:
    def __init__(self,
                 remove_saturation=True,
                 f1=100,
                 f2=400,
                 order=700,
                 peak_min_height=0.1,
                 peak_min_distance=500):
        self.remove_saturation = remove_saturation
        self.f1 = f1
        self.f2 = f2
        self.filter_order = order
        self.peak_min_height = peak_min_height
        self.peak_min_distance = peak_min_distance

    def predict(self,
                ecometry_name,
                compute_speed=True,
                estimated_speed=None,
                use_correlation=False):
        # Load signal.
        df, output_signal_original = load_sample_file(ecometry_name)
        if self.remove_saturation:
            output_signal_original = remove_saturation_method(output_signal_original)

        if use_correlation:
            input_signal_values, _ = get_input_signal(df.Frec1,
                                                      df.Down,
                                                      df.IncF,
                                                      df.Nciclos)
            output_signal = signal.correlate(output_signal_original, input_signal_values)

            # Remove input signal from correlation, plus an arbitrary number of samples.
            output_signal = output_signal[len(input_signal_values) + 10:]
        else:
            output_signal = output_signal_original

        # Normalize.
        output_signal_norm = output_signal / np.max(np.abs(output_signal))

        # Compute signal frequency.
        output_signal_sampling_freq = 930000 / df.Down

        # Band-pass filter
        fir_filter = signal.firwin(self.filter_order + 1, [self.f1, self.f2], pass_zero=False,
                                   fs=output_signal_sampling_freq)
        output_signal_filtered = signal.lfilter(fir_filter, 1, output_signal_norm)

        # Filter signal.
        filtered_signal_envelope = signal.hilbert(output_signal_filtered)
        filtered_signal_envelope = np.abs(filtered_signal_envelope)

        # Compute peaks.
        peaks_ix, peaks = signal.find_peaks(filtered_signal_envelope,
                                            height=self.peak_min_height,
                                            distance=self.peak_min_distance)
        peaks = peaks['peak_heights']
        peaks_ix = peaks_ix / output_signal_sampling_freq

        if compute_speed:
            if len(peaks_ix) > 1:
                peaks_average_diff = np.average(np.diff(peaks_ix))
                estimated_speed = COUPLES_DISTANCE / peaks_average_diff
            else:
                estimated_speed = None

            return estimated_speed
        else:
            if self.remove_saturation:
                _, output_signal_complete = load_sample_file(ecometry_name)
                removed_samples = len(output_signal_complete) - len(output_signal_original)
                removed_samples = removed_samples / output_signal_sampling_freq
            else:
                removed_samples = 0

            final_peaks_ix = []
            final_peaks = []
            for peak_ix, peak in zip(peaks_ix, peaks):
                if (peak_ix + removed_samples) * estimated_speed / 2 >= MIN_DEPTH:
                    final_peaks_ix.append(peak_ix)
                    final_peaks.append(peak)
            peaks_ix = final_peaks_ix
            peaks = final_peaks

            if len(peaks_ix) > 0:
                # peaks_ix = peaks_ix[peaks_ix > 1.0]
                max_ix_value = peaks_ix[np.argmax(peaks)]

                # Add removed samples.
                max_ix_value += removed_samples
                estimated_depth = max_ix_value * estimated_speed / 2
            else:
                estimated_depth = None

            return estimated_depth


class CepstrumEstimation:
    def __init__(self,
                 remove_saturation=True,
                 couplet_separation=COUPLES_DISTANCE,
                 min_speed=200,
                 max_speed=500,
                 f1=100,
                 f2=280,
                 filter_order=200,
                 option=1):
        self.remove_saturation = remove_saturation
        self.couplet_separation = couplet_separation
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.f1 = f1
        self.f2 = f2
        self.filter_order = filter_order
        self.option = option

    def predict(self, ecometry_name):
        # Load signal.
        df, output_signal = load_sample_file(ecometry_name)
        if self.remove_saturation:
            output_signal = remove_saturation_method(output_signal)

        # Normalize.
        output_signal_norm = output_signal / np.max(np.abs(output_signal))

        # Compute signal frequency.
        output_signal_sampling_freq = 930000 / df.Down

        # Get min and max couplet separation.
        max_sep = self.couplet_separation / self.min_speed
        min_sep = self.couplet_separation / self.max_speed

        # Get boundaries in samples.
        initial_position = int(np.fix(min_sep * output_signal_sampling_freq))
        last_position = int(np.ceil(max_sep * output_signal_sampling_freq))

        # Filter signal.
        f1 = self.f1
        f2 = self.f2
        order = self.filter_order
        fir_filter = signal.firwin(order + 1, [f1, f2], pass_zero=False,
                                   fs=output_signal_sampling_freq)
        output_signal_filtered = signal.lfilter(fir_filter, 1, output_signal_norm)

        # Calculate cepstrum from filtered signal.
        rceps = np.real(ifft(np.log(np.abs(fft(output_signal_filtered)))))

        if self.option == 1:
            # max del abs
            pos = np.argmax(np.abs(rceps[initial_position:last_position])) + initial_position
        elif self.option == 2:
            # max
            pos = np.argmax(rceps[initial_position:last_position]) + initial_position
        elif self.option == 3:
            # min
            pos = np.argmin(rceps[initial_position:last_position]) + initial_position

        # Get speed.
        estimated_speed = self.couplet_separation * output_signal_sampling_freq / pos

        return estimated_speed
