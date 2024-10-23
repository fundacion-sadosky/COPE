import numpy as np
import pandas as pd


def load_sample_file(sample_full_path):
    df = pd.read_json(sample_full_path, typ='series')

    final_values = []

    data_values = df.datos
    data_values = data_values.strip("[]").split(",")

    for value in data_values:
        # Convierte el valor hexadecimal a un entero sin signo de 32 bits.
        value = int(value, 16) & 0xFFFFFFFF

        # Verifica si el bit más significativo es 1 para determinar si el valor está en
        # complemento a uno.
        if value & 0x80000000:
            # Si el bit más significativo es 1, convierte el valor a su complemento a dos.
            value = -(0x100000000 - value)

        # Agrega el valor a la lista de enteros.
        final_values.append(value)

    return df, final_values


def remove_saturation(signal_values):
    # Get signal peak positions based on 0.95 quantile.
    peaks = signal_values > np.quantile(signal_values, 0.95)
    saturation_points_ix = np.asarray(peaks).nonzero()[0]

    # Get peak positions diffs and get peak with max diff with the following.
    diffs = np.diff(saturation_points_ix)
    max_diff = np.argmax(diffs)
    max_saturation_point_ix = saturation_points_ix[max_diff]

    # Check if saturation point seems reasonable.
    if max_saturation_point_ix <= 0.5 * len(signal_values):
        signal_values = signal_values[max_saturation_point_ix:]
    else:
        raise Exception("Saturation point is greather than half the length of the signal.")

    return signal_values
