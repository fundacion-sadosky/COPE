import os, glob, fitz
import pandas as pd
import re


def read_dinamometrias(path):
    ''' Read PDF files, extract relevant information and create a CSV file. 

    Params
    ------
    path : str
        Path to directory with files. The directory should have one sub-directory per year, and
        each year sub-directory should also have one sub-directory per month.
    '''

    # Get sub-directories.
    os.chdir(path)
    dirs = os.listdir()
    years = [d for d in dirs if d[0] != '.']

    all_data = []

    for year in years:
        dirs = os.listdir(year)
        months = [m for m in dirs if m[0] != '.']
        for month in months:
            # Get all PDF files inside year-month directory.
            files = glob.glob(os.path.join(year, month) + '/*.pdf')
            for file in files:
                doc = fitz.open(file)  # any supported document type
                if doc.page_count > 3:
                    print(f'Processing file: {file}')
                    page = doc[0]  # we want text from this page

                    data = []
                    date = os.path.basename(file)[:10].replace('203-', '2023-')
                    data.append(date)
                    data.append(' '.join(os.path.basename(file).replace('  ', ' ').split(' ')[1:3]))
                    # porcentaje agua
                    rect = page.get_textbox(fitz.Rect([470, 410, 500, 415])).split(' ')[0]
                    data.append(rect)

                    # presion casing
                    rect = re.findall("\d+\.\d+", page.get_textbox(fitz.Rect([470, 430, 500, 435])))[0]
                    data.append(rect)

                    # presion boca pozo
                    rect = re.findall("\d+\.\d+", page.get_textbox(fitz.Rect([470, 440, 500, 445])))[0]
                    data.append(rect)

                    # yacimiento
                    data.append(page.get_textbox(fitz.Rect(120, 110, 180, 140)).split(' ')[0])

                    # zona
                    data.append(page.get_textbox(fitz.Rect(320, 50, 380, 100)))

                    # carrera
                    data.append(page.get_textbox(fitz.Rect(140, 180, 240, 181)).split(' ')[0])

                    # gpm
                    data.append(page.get_textbox(fitz.Rect(180, 182, 250, 185)).split('\n')[0])

                    page = doc[2]

                    # TV
                    rect = re.findall("\d+", page.get_textbox(fitz.Rect([185, 210, 210, 210])))[0]
                    data.append(rect)

                    # SV
                    rect = re.findall("\d+", page.get_textbox(fitz.Rect([185, 220, 210, 225])))[0]
                    data.append(rect)

                    # CBE
                    rect = re.findall("\d+", page.get_textbox(fitz.Rect([185, 230, 210, 235])))[0]
                    data.append(rect)

                    # TMax
                    rect = re.findall("\d+", page.get_textbox(fitz.Rect([185, 240, 210, 245])))[0]
                    data.append(rect)

                    # VPmax
                    rect = re.findall("\d+\.\d+", page.get_textbox(fitz.Rect([185, 250, 210, 255])))[0]
                    data.append(rect)

                    # Carga Max. Fondo
                    rect = re.findall("\d+", page.get_textbox(fitz.Rect([185, 260, 210, 265])))[0]
                    data.append(rect)

                    # Carga Min. Fondo
                    rect = re.findall("-?\d+", page.get_textbox(fitz.Rect([185, 270, 210, 275])))[0]
                    data.append(rect)

                    # Sobre recorrido
                    rect = re.findall("\d+\.\d+", page.get_textbox(fitz.Rect([185, 280, 210, 285])))[0]
                    data.append(rect)

                    # Estiramiento
                    rect = re.findall("\d+\.\d+", page.get_textbox(fitz.Rect([185, 290, 210, 295])))[0]
                    data.append(rect)

                    page = doc[3]
                    # profundidad bomba
                    rect = page.get_textbox(fitz.Rect([150, 190, 250, 219]))
                    data.append(rect)

                    # cuplas
                    rect = page.get_textbox(fitz.Rect([150, 270, 250, 275]))
                    data.append(rect)

                    # produccion petroleo
                    rect = page.get_textbox(fitz.Rect([150, 305, 250, 310]))
                    data.append(rect)

                    # profundidad ga
                    rect3 = page.get_textbox(fitz.Rect([490, 120, 550, 165])).split('\n')
                    if rect3[0] != 'Indeterminado':
                        profundidad_gas = rect3[0]
                        data.append(profundidad_gas)
                        profundidad_niveles = rect3[1]
                        data.append(profundidad_niveles)
                        velocidad_sonido = rect3[2]
                        data.append(velocidad_sonido)
                        sumergencia_efectiva = rect3[3]
                        data.append(sumergencia_efectiva)
                        presion_dinamica = rect3[4]
                        data.append(presion_dinamica)
                        presion_interface = rect3[5]
                        data.append(presion_interface)
                        all_data.append(data)
                    else:
                        for i in list(range(6)):
                            data.append(None)

    df = pd.DataFrame(all_data,
                      columns=['fecha', 'pozo', 'porcentaje agua',
                               'presion casing', 'presion boca pozo',
                               'yacimiento', 'zona',
                               'carrera', 'gpm',
                               'tv', 'sv', 'cbe', 'tmax', 'vpmax',
                               'carga max fondo', 'carga min fondo',
                               'sobre recorrido', 'estiramiento',
                               'profundidad bomba',
                               'cuplas', 'produccion petroleo', 'profundidad gas',
                               'profundidad niveles', 'velocidad sonido', 'sumergencia efectiva',
                               'presion dinamica', 'presion interface'])

    df.to_csv('dinamometrias.csv', index=False)
