import pandas as pd
import os

def cargar_datos():
    # 1. Ruta absoluta del directorio donde se encuentra este script (src)

    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    print(f'ruta actual del script: {ruta_actual}')

    #2. Subir un nivel para llegar a la carpeta donde esta la base de datos

    ruta_proyecto = os.path.dirname(ruta_actual)
    print(f'ruta del proyecto: {ruta_proyecto}')

    #3. Construir la ruta completa del archivo Excel

    ruta_excel = os.path.join(ruta_proyecto, 'Base_de_datos.xlsx')

    #4. Leemos el archivo Excel y lo leemos

    df = pd.read_excel(ruta_excel)
    print('datos cargados del archivo Excel:')
    print(df.head())


    return df

if __name__ == '__main__':
    datos = cargar_datos()
    print('Datos cargados correctamente')
    print('Muestra de los datos:')
    print(datos.head())
