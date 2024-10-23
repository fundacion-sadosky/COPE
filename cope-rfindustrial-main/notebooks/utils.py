def compute_estimations_all_ecometries(estimation_method, estimation_method_name):
    folders = glob(DATA_PATH + '*')
    equipments = [c for c in folders if len(os.path.basename(c)) < 4]
    
    results = {}
    
    for equipment in equipments:
        equipment_trials = [t for t in glob(equipment + '/_*')]
        for trial in equipment_trials:
            ecometries = glob(trial + '/*')
            for ecometry in ecometries:
                df, output_signal = load_sample_file(ecometry)
                estimation = estimation_method(df, output_signal)
                results[ecometry] = estimation
    
    return results

def am_demodulation(df, signal_values):
    # Normalize to avoid precision issues.
    signal_norm = signal_values / np.max(np.abs(signal_values))

    # Compute sampling frequency and get time ticks.
    signal_sampling_freq = 930000 / df.Down
    x_values = np.linspace(0, len(signal_values) / signal_sampling_freq, len(signal_values))

    # Filter signals.
    f1, f2 = 100, 200
    order = 400
    fir_filter = signal.firwin(order + 1, [f1, f2], pass_zero=False, fs=signal_sampling_freq)
    signal_filtered = lfilter(fir_filter, 1, signal_norm)     

    # Get signal abs.
    signal_abs = np.abs(signal_filtered)

    # Low pass filter.
    b, a = butter(2, 70, btype='low', analog=False, fs=signal_sampling_freq)
    signal_filtered_low_pass = filtfilt(b, a, signal_abs)

    # Find peaks.
    x_time_limit = 0.2
    time_limited_signal = signal_filtered_low_pass[x_values >= x_time_limit]
    peaks_ix, peaks = signal.find_peaks(signal_filtered_low_pass, height=0.5 * max(time_limited_signal), distance=500)
    peaks = peaks['peak_heights']
    peaks_filter = peaks_ix > (x_time_limit * signal_sampling_freq)
    peaks_ix = peaks_ix[peaks_filter] / signal_sampling_freq
    peaks = peaks[peaks_filter]

    if len(peaks_ix) > 0:
        peaks_average_diff = np.average(np.diff(peaks_ix))
        estimated_speed = 9.6 / peaks_average_diff
    else:
        return None

    return estimated_speed