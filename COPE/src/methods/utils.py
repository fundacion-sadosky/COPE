import os
import struct

import pandas as pd

from src.data.settings import DATA_PATH
from src.methods.settings import RESULTS_STORE


def compute_stats_for_method(method_instance, well_ids=None, verbose=1, data_path=None):
    results = []

    if data_path is None:
        root_paths = os.listdir(DATA_PATH)
    else:
        root_paths = os.listdir(data_path)

    for path in root_paths:
        # Filter other directories.
        if len(path) < 3:
            assert type(well_ids) == list, 'wells ids must be a list'

            # Check if well is included in well ids.
            if (well_ids is None) or (well_ids and int(path) in well_ids):
                experiments_paths = os.listdir(os.path.join(DATA_PATH, path))
                for experiment in experiments_paths:
                    # Filter non-experiment directories.
                    if '_Frec1' in experiment:
                        echometry_paths = os.listdir(os.path.join(DATA_PATH, path, experiment))
                        for echometry in echometry_paths:
                            echometry_path = os.path.join(DATA_PATH, path, experiment, echometry)
                            try:
                                estimated_speed = method_instance.predict(echometry_path)
                            except Exception as e:
                                if verbose == 1:
                                    print('Failed to predict file: ', echometry_path)
                                    print(str(e))
                                estimated_speed = None
                            results.append([path, experiment, echometry, estimated_speed])

    df = pd.DataFrame(results,
                      columns=['id_pozo', 'experimento', 'ecometria', 'velocidad_estimada'])
    return df


def compute_error(df_predictions, well_id, real_speed):
    _df = df_predictions[df_predictions.id_pozo.astype(int) == well_id]
    error = real_speed - _df.velocidad_estimada.dropna().mean()
    print(f'El error aproximado es de {round(error, 4)} m/s')
    return error


def save_results(method_name,
                 df_estimations):
    file_path = RESULTS_STORE.replace('.csv', f'_{method_name}.csv')
    if os.path.isfile(file_path):
        df = pd.read_csv(file_path, sep='\t')
    else:
        df = pd.DataFrame(
            columns=['id_pozo',
                     'experimento',
                     'ecometria',
                     'velocidad_estimada',
                     'timestamp',
                     'parametros_metodo']
        )

    df = pd.concat([df_estimations,
                    df], ignore_index=True)

    df.to_csv(file_path, index=None, sep='\t')

    return df


def get_input_signal(initial_freq, signal_step, freq_step, n_cycles):
    with open('tabla_seno.bin', 'rb') as file:
        bytes = file.read()
        sine_values = struct.unpack('{}H'.format(len(bytes) // 2), bytes)

    # Se crea una lista vacía que se llenará con los valores de la señal muestreada
    signal_values = []
    cycles_info = []

    # Se inicializan las variables de iteración
    inter = 0
    freq = initial_freq
    step = signal_step // 32

    last_cycle_inter = 0

    # Se ejecuta un ciclo para generar la señal senoidal
    for i in range(n_cycles):
        j = 0
        # Se recorre la tabla de valores precalculados de la función seno
        while j < len(sine_values):
            # Se verifica si se debe tomar una muestra en este punto
            if inter % step == 0:
                # Se obtiene el valor de la función seno en este punto
                y = sine_values[j]
                # Se agrega el valor de la muestra a la lista
                signal_values.append(y)
            # Se avanza al siguiente punto de la señal
            j += freq
            # Se incrementa el índice que se utiliza para verificar si se debe tomar una muestra
            inter += 1
        # Se recorre la tabla de valores precalculados de la función seno en sentido inverso
        while j < 2 * len(sine_values):
            # Se verifica si se debe tomar una muestra en este punto
            if inter % step == 0:
                # Se obtiene el valor de la función seno en este punto y se invierte su signo
                y = -sine_values[j - len(sine_values)]
                # Se agrega el valor de la muestra a la lista
                signal_values.append(y)
            # Se avanza al siguiente punto de la señal
            j += freq
            # Se incrementa el índice que se utiliza para verificar si se debe tomar una muestra
            inter += 1
        # Se ajusta el índice para reiniciar la generación de la señal en la tabla de valores
        j -= 2 * len(sine_values)

        cycles_info.append((inter - last_cycle_inter, freq))
        last_cycle_inter = inter

        # Se incrementa la frecuencia de la señal para el siguiente ciclo
        freq += freq_step

    return signal_values, cycles_info
