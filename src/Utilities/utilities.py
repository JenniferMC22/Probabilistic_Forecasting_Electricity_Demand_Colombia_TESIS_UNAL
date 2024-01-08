import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import plotly.express as px
from pandas.api.types import CategoricalDtype
import pyarrow.parquet as pq
import datetime as dt
from tqdm import tqdm_notebook
from itertools import product
from pmdarima import auto_arima
import os
import warnings

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
pd.options.display.float_format = '{:.2f}'.format



#sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split,cross_val_score,RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error,mean_squared_error, mean_absolute_percentage_error, r2_score, median_absolute_error
from sklearn.metrics import mean_pinball_loss

#scipy
from scipy.stats import kruskal
from scipy.stats import friedmanchisquare
from scipy.signal import periodogram
from scipy.stats import norm
from scipy.stats.contingency import association
from scipy.stats import chi2_contingency
from scipy.stats import chi2
from scipy import stats


#statsmodels
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import rmse
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


#keras
from keras.models import Sequential
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch
from keras.layers import SimpleRNN, LSTM, GRU, Dropout, Dense, Bidirectional





class demand_utilities: 
    def __init__(self):
        """
        Class constructor demand 
        """
        print("Class demand_utilities...", self)

    def read_demand_data(ruta):
        """
        Esta función lee los archivos de Excel de demanda de energía en el sistema interconectado nacional (SIN) de una ruta determinada y devuelve una única tabla consolidada.
        
        Parámetros:
        - ruta: Ruta donde se encuentran los archivos de Excel que contienen la información de demanda de energía.
        
        Resultado:
        - Una tabla consolidada con la información de demanda de energía de todos los archivos de Excel encontrados en la ruta especificada.
        """
        files = os.listdir(ruta)
        filenames = [f for f in files if f.startswith("Demanda_Energia_SIN_")]
        dfs = []
        for filename in filenames:
            if filename.endswith(".xlsx"):
                df = pd.read_excel(os.path.join(ruta, filename))
            elif filename.endswith(".xls"):
                df = pd.read_excel(os.path.join(ruta, filename), engine="xlrd")
            else:
                continue
            dfs.append(df)
        demand = pd.concat(dfs, ignore_index=True)
        return demand
    
    def import_demand(ruta_directorio):
        carpetas = [carpeta for carpeta in os.listdir(ruta_directorio) if os.path.isdir(os.path.join(ruta_directorio, carpeta))]
        datos_hoja_real = []

        for carpeta in carpetas:
            ruta_carpeta = os.path.join(ruta_directorio, carpeta)
            archivos_xls = [archivo for archivo in os.listdir(ruta_carpeta) if archivo.endswith('.xls') or archivo.endswith('.xlsx')]

            for archivo in archivos_xls:
                ruta_archivo = os.path.join(ruta_carpeta, archivo)
                xls = pd.ExcelFile(ruta_archivo)
                hojas_disponibles = xls.sheet_names

                if 'real' in hojas_disponibles:
                    datos = pd.read_excel(xls, sheet_name='real')
                    datos_hoja_real.append(datos)

        return datos_hoja_real


    
    
    def filtro_fechas(fila):
        """
        Esta función filtra las filas de un DataFrame que no contengan una fecha válida en la primera columna y una cantidad numérica en la segunda columna.
        
        Parámetros:
        - fila: Fila del DataFrame que se va a filtrar.
        
        Resultado:
        - True si la fila cumple con las condiciones de tener una fecha válida en la primera columna y una cantidad numérica en la segunda columna, False en caso contrario.
        """
        # Definir la función para filtrar las filas
        # Comprobar si la fila es NaN
        if pd.isna(fila[0]) or pd.isna(fila[1]):
            return False
        # Comprobar si la fila es una fecha y cumple con el patrón YYYY-MM-DD
        try:
            pd.to_datetime(fila[0], format='%Y-%m-%d')
        except ValueError:
            return False
        # Si pasa todas las comprobaciones, devuelve True para mantener la fila
        return True
    
    def datos_diarios_a_mensuales(datos_diarios, columna_demanda,agregation_operation):
        """
        Convierte un dataframe de datos diarios a datos mensuales, agregando los valores diarios 
        para cada mes y eliminando los días que no corresponden a un mes completo. 

        Parámetros
        ----------
        datos_diarios : DataFrame
            DataFrame con los datos diarios.
        columna_demanda : str
            Nombre de la columna que contiene los valores de la demanda.
            
        Retorna
        -------
        DataFrame
            DataFrame con los datos mensuales.
        """
        # Agregar columna de día
        datos_diarios['Día'] = pd.to_datetime(datos_diarios['Fecha']).dt.day
        
        # Obtener año y mes
        datos_diarios['Año'] = pd.DatetimeIndex(datos_diarios['Fecha']).year
        datos_diarios['Mes'] = pd.DatetimeIndex(datos_diarios['Fecha']).month
        
        # Agregar columna de fecha
        datos_diarios['Fecha'] = pd.to_datetime(datos_diarios[['Año', 'Mes', 'Día']].apply(lambda x: '/'.join(x.astype(str)), axis=1), format='%Y/%m/%d')

        
        # Eliminar días sobrantes al final de cada mes
        datos_diarios = datos_diarios[datos_diarios['Día'] <= pd.to_datetime(datos_diarios[['Año', 'Mes', 'Día']].apply(lambda x: '/'.join(x.astype(str)), axis=1)).dt.days_in_month]

        
        # Agrupar por año y mes y promediar los valores diarios
        datos_mensuales = datos_diarios.groupby(['Año', 'Mes'], as_index=False).agg({columna_demanda: agregation_operation})
        
        return datos_mensuales
    
    def filtrar_exportar_parquet(ruta, fecha_inicio, fecha_fin, nombre_archivo, startwith):
        """
        Filtra los archivos .data en la ruta especificada por fecha y exporta el dataframe resultante en formato parquet.

        Parámetros
        ----------
        ruta : str
            Ruta donde se encuentran los archivos .data.
        fecha_inicio : str
            Fecha de inicio del filtro en formato 'YYYY-MM-DD'.
        fecha_fin : str
            Fecha de fin del filtro en formato 'YYYY-MM-DD'.
        nombre_archivo : str
            Nombre del archivo parquet resultante sin la extensión.

        Retorna
        -------
        None
            Exporta el dataframe resultante en formato parquet en la misma ruta especificada.
        """
        # Obtener lista de archivos .data en la ruta dada
        #archivos = [archivo for archivo in os.listdir(ruta) if archivo.endswith('.data')]
        # Obtener lista de archivos .data en la ruta dada que cumplan la condición de que el nombre empiece con "TSSM_CON@"
        archivos = [archivo for archivo in os.listdir(ruta) if archivo.endswith('.data') and archivo.startswith(startwith)]
        #TSSM_CON@
        #PTPM_CON_INTER@


        # Crear una lista vacía para almacenar los dataframes de cada archivo
        lista_df = []

        # Iterar sobre los archivos y leerlos como dataframes de pandas
        for archivo in archivos:
            ruta_archivo = os.path.join(ruta, archivo)
            df = pd.read_csv(ruta_archivo, delimiter="|", parse_dates=["Fecha"])
            # Filtrar por fecha
            df = df[(df["Fecha"] >= fecha_inicio) & (df["Fecha"] <= fecha_fin)]
            lista_df.append(df)

        # Concatenar los dataframes por filas
        df_final = pd.concat(lista_df, axis=0)

        # Exportar dataframe resultante en formato parquet
        ruta_archivo_parquet = os.path.join(ruta, f"{nombre_archivo}.parquet")
        df_final.to_parquet(ruta_archivo_parquet)

    def read_parquet_file_and_resample(filepath:str, column:str, resampling:str)-> pd.DataFrame:
        """
        Esta función carga un archivo en formato parquet como un DataFrame de Pandas, convierte una columna específica a tipo datetime, 
        establece esa columna como índice, agrupa los datos por el intervalo de tiempo especificado y calcula la media de los datos 
        en ese intervalo.

        Args:
        - filepath (str): Ruta del archivo .parquet que se desea leer.
        - column (str): Nombre de la columna que se desea convertir a tipo datetime y establecer como índice.
        - resampling (str): Intervalo de tiempo en el que se desea agrupar los datos. Se debe especificar usando una cadena de texto 
                            en formato de timedelta de Pandas (por ejemplo: "D" para días, "H" para horas, "30T" para intervalos de 30 minutos).

        Returns:
        - df_resampled (pd.DataFrame): DataFrame de Pandas con los datos resampleados.

        """
        # Leer el archivo parquet como un dataframe de pandas
        df = pd.read_parquet(filepath)
        
        # Convertir la columna especificada a tipo datetime
        df[column] = pd.to_datetime(df[column])
        
        # Establecer la columna especificada como índice
        df = df.set_index(column)
        
        # Agrupar por el intervalo de tiempo especificado y calcular la media de la columna "Valor"
        df_resampled = df.resample(resampling).mean()
        
        return df_resampled
    
    def comparar_dataframes(df1, df2):
        """
        Compara los valores de dos dataframes y retorna True si son iguales y False en caso contrario.

        Parámetros
        ----------
        df1 : DataFrame
            Primer dataframe a comparar.
        df2 : DataFrame
            Segundo dataframe a comparar.

        Retorna
        -------
        bool
            Retorna True si los dataframes son iguales y False en caso contrario.
        """
        # Comparar tamaños de los dataframes
        if df1.shape != df2.shape:
            return False

        # Comparar los valores de cada celda
        comparison = df1.values == df2.values
        equal_arrays = comparison.all()
        if equal_arrays:
            return True
        else:
            # Encontrar las posiciones de las celdas que son diferentes
            pos = np.where(comparison == False)
            print("Las siguientes celdas son diferentes:")
            for row, col in zip(pos[0], pos[1]):
                print(f" - Fila {row}, columna {col}: {df1.iat[row, col]} != {df2.iat[row, col]}")
            return False

    def import_parquet_with_date(filepath, start_date, freq):
    # Importar archivo parquet
        table = pq.read_table(filepath)
        df = table.to_pandas()

        # Crear fecha inicial
        date = pd.date_range(start_date, periods=len(df), freq=freq)
        
        # Agregar fecha al DataFrame
        df['fecha'] = date
        
        return df
    
    def check_dataframe(df, fecha_col='fecha', valor_col='Valor'):
        """
        Esta función comprueba si un DataFrame de serie de tiempo cumple con los requisitos necesarios para su análisis.

        Parámetros:
        - df: DataFrame que contiene la serie de tiempo a comprobar.
        - fecha_col: nombre de la columna que contiene la fecha. Por defecto, 'fecha'.
        - valor_col: nombre de la columna que contiene los valores. Por defecto, 'Valor'.

        Resultado:
        - None. La función imprime la frecuencia de la serie de tiempo y el rango de fechas si se cumple con los requisitos, de lo contrario imprime un mensaje de error.
        """
        # Chequeamos si la columna de fecha existe y es de tipo datetime
        if fecha_col.lower() in [c.lower() for c in df.columns] and isinstance(getattr(df, fecha_col.lower())[0], pd.Timestamp):
            # Calculamos la frecuencia de la columna fecha
            freq = pd.infer_freq(getattr(df, fecha_col.lower()))
            print('Frecuencia:', freq)
            # Calculamos el rango de fechas
            min_date = getattr(df, fecha_col.lower()).min()
            max_date = getattr(df, fecha_col.lower()).max()
            print('Rango de fechas:', min_date, '-', max_date)
            # Chequeamos si existen las columnas de año, mes y valor
        elif {'Año', 'Mes', valor_col.lower()} <= set([c.lower() for c in df.columns]):
            # Unimos las columnas de año y mes para obtener la columna de fecha
            df[fecha_col] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'].astype(str), format='%Y-%m')
            # Calculamos la frecuencia de la columna fecha
            freq = pd.infer_freq(df[fecha_col])
            print('Frecuencia:', freq)
            # Calculamos el rango de fechas
            min_date = df[fecha_col].min()
            max_date = df[fecha_col].max()
            print('Rango de fechas:', min_date, '-', max_date)
        else:
            print('El dataframe no tiene una columna de fecha o las columnas de año, mes y', valor_col)

    def filter_dataframe_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Filtra un dataframe por un rango de fechas.

        Args:
            df (pd.DataFrame): El dataframe a filtrar.
            start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
            end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.

        Returns:
            pd.DataFrame: Un nuevo dataframe que contiene solo las filas que cumplen con el rango de fechas especificado.
        """
        mask = (df['Fecha'] >= start_date) & (df['Fecha'] <= end_date)
        filtered_df = df.loc[mask]
        return filtered_df
    
    # Verificar que el índice temporal está completo
    # ==============================================================================
    def complete_serie(df: pd.DataFrame) -> bool:
        """
        Esta función comprueba si una serie de tiempo tiene todos los valores de fecha en el rango esperado para su frecuencia.

        Parámetros:
        - df: DataFrame que contiene la serie de tiempo a comprobar.

        Resultado:
        - True si la serie de tiempo tiene todos los valores de fecha en el rango esperado para su frecuencia, False en caso contrario.
        """
        # Comprobar si los valores de índice de la serie de tiempo están en el rango esperado
        return (df.index == pd.date_range(start=df.index.min(),
                                        end=df.index.max(),
                                        freq=df.index.freq)).all()
    
    import pandas as pd

    def crear_columna_fecha(datos, ano_col='Año', mes_col='Mes', fecha_col='fecha', y_month_col='y-month'):
        """
        Crea una columna de fecha en un DataFrame a partir de columnas de año y mes existentes.

        Parámetros:
        - datos: DataFrame que contiene las columnas de año y mes.
        - ano_col: Nombre de la columna de año (por defecto: 'Año').
        - mes_col: Nombre de la columna de mes (por defecto: 'Mes').
        - fecha_col: Nombre para la nueva columna de fecha (por defecto: 'fecha').
        - y_month_col: Nombre para la nueva columna de fecha formateada (por defecto: 'y-month').

        Resultado:
        - DataFrame con la nueva columna de fecha y la columna de fecha formateada.
        """

        # Crear la columna de fecha
        datos[fecha_col] = pd.to_datetime(datos[ano_col].astype(str) + '-' + datos[mes_col].astype(str), format='%Y-%m')

        # Formatear la fecha
        datos[y_month_col] = datos[fecha_col].dt.strftime('%Y-%m')

        return datos
    
    def test_stationarity(dataframe):
        results = {}
        for column in dataframe.columns:
            # realiza la prueba ADF para la serie de tiempo actual
            result = adfuller(dataframe[column])
            # guarda los resultados de la prueba para la serie de tiempo actual
            results[column] = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'used_lags': result[2],
                'n_observations': result[3],
                'critical_values': result[4]
            }
        return results
    
    def time_series_tests(df, column_name):
        """
        Función que realiza pruebas estadísticas para analizar la estacionariedad y tendencia de una serie de tiempo.

        Parámetros:
        df (pandas.DataFrame): el dataframe que contiene la serie de tiempo.
        column_name (str): el nombre de la columna que contiene la serie de tiempo.

        Retorna:
        dict: un diccionario que contiene los resultados de las pruebas estadísticas realizadas.
        Cada prueba tiene un nombre, un resultado y una inferencia asociada. El diccionario tiene la siguiente estructura:
        {
            'Prueba 1': {
                'nombre_prueba': str,
                'resultado': {
                    'valor_p': float,
                    'otro_resultado': float
                },
                'inferencia': str
            },
            'Prueba 2': {
                'nombre_prueba': str,
                'resultado': {
                    'valor_p': float,
                    'otro_resultado': float
                },
                'inferencia': str
            },
            ...
        }
        """
        
        # Extraer la serie de tiempo de la columna especificada en el dataframe
        time_series = df[column_name]

        # Prueba de la pendiente de regresión (si hay suficientes observaciones y suficiente variabilidad)
        if len(time_series) >= 10 and time_series.std() > 0:
            try:
                slope, intercept, r_value, p_value, std_err = sm.OLS(time_series, sm.add_constant(time_series.index)).fit().params
                regression_test = {'nombre_prueba': 'Prueba de la pendiente de regresión',
                                'resultado': {'valor_p': p_value, 'pendiente': slope, 'intercepto': intercept},
                                'inferencia': 'Si el valor p es menor que 0.05, se puede rechazar la hipótesis nula de que la pendiente es cero, indicando una tendencia significativa.'}
            except ValueError:
                regression_test = {'nombre_prueba': 'Prueba de la pendiente de regresión',
                                'resultado': None,
                                'inferencia': 'No se pudieron ajustar los coeficientes de la regresión debido a la falta de variabilidad en la serie de tiempo.'}
        else:
            regression_test = {'nombre_prueba': 'Prueba de la pendiente de regresión',
                            'resultado': None,
                            'inferencia': 'No hay suficientes observaciones o variabilidad en la serie de tiempo para realizar esta prueba.'}


        # Prueba de la media móvil
        rolling_mean = time_series.rolling(window=12).mean()
        rolling_mean_test = {'nombre_prueba': 'Prueba de la media móvil',
                            'resultado': {'valor_p': adfuller(rolling_mean.dropna())[1]},
                            'inferencia': 'Si el valor p es menor que 0.05, se puede rechazar la hipótesis nula de que la serie no tiene una tendencia significativa.'}

        # Prueba de Mann-Kendall
        trend = pd.Series(time_series).cummax().astype(float)
        mk_result = pd.Series([trend.autocorr() * 2 / (len(time_series) - 1)], index=['MKScore'], name='MKScore')
        mk_test = {'nombre_prueba': 'Prueba de Mann-Kendall',
                'resultado': {'MKScore': mk_result[0]},
                'inferencia': 'Si el valor MKScore es mayor que 0 y menor que 0.25, la serie tiene una tendencia creciente con un nivel de confianza del 95%. Si el valor es menor que 0 y mayor que -0.25, la serie tiene una tendencia decreciente con un nivel de confianza del 95%.'}

        # Prueba de la raíz unitaria
        adf_result = adfuller(time_series)
        adf_test = {'nombre_prueba': 'Prueba de la raíz unitaria',
                    'resultado': {'valor_p': adf_result[1], 'valores_críticos': adf_result[4]},
                    'inferencia': 'Si el valor p es menor que 0.05, se puede rechazar la hipótesis nula de que la serie de tiempo tiene una raíz unitaria, lo que indica una tendencia estacionaria.'}

        # Devolver el diccionario con los resultados

        test_results = {'Prueba de la pendiente de regresión': regression_test,
                        'Prueba de la media móvil': rolling_mean_test,
                        'Prueba de Mann-Kendall': mk_test,
                        'Prueba de raíz unitaria': adf_test}
        
        return test_results
    
    def exploratory_time_series_analysis(df, freq):
        """
        Función que realiza un análisis exploratorio inicial de un dataframe que contiene series de tiempo
        
        Parámetros:
        - df: Dataframe que contiene las series de tiempo a analizar
        - freq: Frecuencia de las series de tiempo. Puede ser 'D', 'M', 'Y', etc.
        
        Retorna:
        - Gráficos de las series de tiempo y sus estadísticas descriptivas
        """
        
        # Seleccionar todas las columnas que no sean la primera
        columns = df.columns[1:]
        
        # Crear gráfico de líneas para cada serie de tiempo
        plt.figure(figsize=(16,8))
        for i in range(1, len(columns)):  # Comenzar desde la segunda columna
            plt.subplot(2, int(np.ceil(len(columns)/2)), i)
            plt.plot(df.iloc[:, i])
            plt.xlabel("Índice de tiempo")
            plt.ylabel(df.columns[i])
            plt.title('Serie de tiempo: '+df.columns[i])
        plt.tight_layout()
        plt.show()

        # Descripción estadística de las series de tiempo
        display(df.describe())

        # Histogramas de las series de tiempo
        df.iloc[:, 1:].hist(figsize=(16, 8), bins=20)
        plt.tight_layout()
        plt.show()

        # Gráfico de densidad de las series de tiempo
        df.iloc[:, 1:].plot(kind='kde', figsize=(18, 5))
        plt.xlabel('Valor')
        plt.title('Gráfico de densidad')
        plt.tight_layout()
        plt.show()

        # Matriz de correlación entre las series de tiempo
        corr = df.iloc[:, 1:].corr()
        sns.heatmap(corr, annot=True)
        plt.title('Matriz de correlación')
        plt.tight_layout()
        plt.show()

        # Gráfico de autocorrelación de cada serie de tiempo
        plt.figure(figsize=(12,6))
        for i in range(len(columns)):
            serie = pd.Series(df.iloc[:, i+1].values, index=df.iloc[:, 0])
            pd.plotting.autocorrelation_plot(serie)
            plt.xlabel('Retardo')
            plt.ylabel('Autocorrelación')
            plt.title('Gráfico de autocorrelación: '+df.columns[i+1])
            plt.show()

    def plot_pacf_subplots(dataframe, columns, lags=60, figsize=(10, 15)):
        fig, axs = plt.subplots(nrows=len(columns), figsize=figsize)
        
        for i, col in enumerate(columns):
            plot_pacf(dataframe[col], ax=axs[i], lags=lags)
            axs[i].set_title(col)
            
        plt.show()

    def plot_acf_subplots(dataframe, columns, lags=100, figsize=(10, 15)):
        fig, axs = plt.subplots(nrows=len(columns), figsize=figsize)
        
        for i, col in enumerate(columns):
            plot_acf(dataframe[col], ax=axs[i], lags=lags)
            axs[i].set_title(col)
            
        plt.show()

    def mk_test(x: np.ndarray, alpha: float = 0.05) -> tuple:
        """
        Computes the Mann-Kendall test for trend in a time series.

        Parameters:
        -----------
        x: np.ndarray
            The input time series.
        alpha: float
            The significance level for the hypothesis test (default: 0.05).

        Returns:
        --------
        p: float
            The two-tailed p-value of the hypothesis test.
        h: bool
            Whether or not the null hypothesis is rejected. Returns True if rejected, False otherwise.
        """
        n = len(x)
        s = 0
        for k in range(n-1):
            for j in range(k+1, n):
                s += np.sign(x[j] - x[k])
        var_s = (n*(n-1)*(2*n+5))/18
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        p = 2*(1-norm.cdf(abs(z)))
        h = abs(z) > norm.ppf(1-alpha/2)
        return p, h
    
    def check_zeros_in_column(df, column_name):
        """
        Verifica si hay valores iguales a cero en alguna de las filas de una columna de un dataframe.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe con la información.
        column_name : str
            Nombre de la columna del dataframe que se quiere verificar.
            
        Returns
        -------
        bool
            True si hay valores iguales a cero en alguna de las filas de la columna especificada, False de lo contrario.
        """
        
        # Seleccionar la columna especificada
        column = df[column_name]
        
        # Verificar si hay valores iguales a cero
        return (column == 0).any()
    
    def tsplot(y, lags=None, figsize=(12, 7), style='bmh'):
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
            
        with plt.style.context(style):    
            fig = plt.figure(figsize=figsize)
            layout = (2, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (1, 0))
            pacf_ax = plt.subplot2grid(layout, (1, 1))
            
            y.plot(ax=ts_ax)
            p_value = sm.tsa.stattools.adfuller(y)[1]
            ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
            smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
            smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
            plt.tight_layout()

    def generar_base_rezagos(data, rezagos):
        # Iterador: agregar rezago i-ésimo
        for i in range(1, rezagos + 1):
            # Nombre de la variable del rezago i-ésimo
            nombre_columna = f"Demanda_SIN_{i}"
            # Agregar columna a la base de datos
            data[nombre_columna] = data['Demanda_SIN'].shift(i)

        return data







class StationarityUtilities:
    """
    Clase con utilidades para el análisis de estacionariedad en series de tiempo.
    """
    def __init__(self):
        """
        Constructor de la clase StationarityUtilities.
        """
        print("Class StationarityUtilities...", self)
    @staticmethod
    def kruskalWallisTest(df, column_name, month_col, alpha=0.05):
        """
        Realiza una prueba de Kruskal-Wallis para evaluar si existen diferencias significativas entre los grupos 
        mensuales de una variable específica de un dataframe.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Dataframe con la información de los datos.
        column_name : str
            Nombre de la columna del dataframe que se quiere evaluar.
        month_col : str
            Nombre de la columna del dataframe que contiene la información de los meses.
        alpha : float, optional
            Nivel de significancia deseado para la prueba. El valor por defecto es 0.05.
            
        Returns
        -------
        dict
            Un diccionario con la información del resultado de la prueba.
                'nombre_prueba': 'Kruskal-Wallis'.
                'resultado': Un diccionario con el valor p obtenido.
                'inferencia': Una cadena de texto que indica la inferencia del resultado obtenido.
                    Si el valor p es menor que alpha, se puede rechazar la hipótesis nula de que no hay 
                    diferencias significativas entre los grupos. Si el valor p es mayor o igual que alpha, 
                    no se puede rechazar la hipótesis nula.
        """
        
        months = np.sort(df[month_col].unique())
        samples = [df.loc[df[month_col]==month, column_name] for month in months]
        p, h = kruskal(*samples)
        return {'nombre_prueba': 'Kruskal-Wallis',
                'resultado': {'valor_p': p},
                'inferencia': 'Si el valor p es menor que 0.05, se puede rechazar la hipótesis nula de que no hay diferencias significativas entre los grupos.'}
    @staticmethod
    def TimeSeriesDecomposition(dataframe, column, freq=None):
        """
        Decompose a time series into its trend, seasonality and residuals components
        and plot them.

        Parameters:
        dataframe (pandas.DataFrame): The dataframe containing the time series to decompose.
        column (str): The column of the dataframe that contains the time series to decompose.
        freq (str, optional): The frequency of the time series. If not provided, pandas will try
                            to infer the frequency.

        Returns:
        None
        """

        ts = dataframe.set_index('fecha')[column]
        if freq:
            ts.index = pd.date_range(start=ts.index[0], periods=len(ts), freq=freq)
        decomposition = sm.tsa.seasonal_decompose(ts, model='additive')
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(ts, label='Original', linewidth=1)
        ax.plot(decomposition.trend, label='Trend', linewidth=2)
        ax.plot(decomposition.seasonal, label='Seasonality', linewidth=2)
        ax.plot(decomposition.resid, label='Residuals', linewidth=1)
        ax.legend(loc='best')
        plt.show()

    @staticmethod
    def kruskalWallisSeasonality_test(time_series,freq):
        """
        Kruskal-Wallis test for seasonality
        
        Parameters:
        -----------
        time_series : numpy array or pandas series
            The time series data to be tested
        
        Returns:
        --------
        result : tuple
            A tuple containing the test statistic and p-value
        """
        n = len(time_series)
        # Divide la serie en grupos iguales de acuerdo a la frecuencia
        groups = np.split(time_series, n // freq)
        # Aplica el test de Kruskal-Wallis a los grupos
        statistic, pvalue = kruskal(*groups)

        # Interpret results
        if pvalue < 0.05:
            inference = "La serie de tiempo tiene evidencia de estacionalidad."
        else:
            inference = "La serie de tiempo no tiene evidencia de estacionalidad."
            
        return statistic, pvalue, inference
    @staticmethod
    def FriedmanSeasonality_test(dataframe, column):
        '''    
        Realiza la prueba de sazonalidad de Friedman para una serie de tiempo dada en un DataFrame de pandas.

        Args:
        - dataframe: DataFrame de pandas que contiene la serie de tiempo.
        - column: String que indica el nombre de la columna que contiene la serie de tiempo.

        Returns:
        - f_value: El valor del estadístico de prueba.
        - p_value: El valor p de la prueba.
        - inferencia: Una cadena de texto que indica si hay patrones estacionales significativos o no.

        Ejemplo:
        f, p, inf = friedman_seasonality_test(data, 'columna_de_tiempo')
        print(f"El valor del estadístico de prueba es: {f}")
        print(f"El valor p de la prueba es: {p}")
        print(inf)
        '''
        ts = dataframe.set_index('fecha')[column]
        seasons = np.unique(ts.index.strftime('%B'))
        seasons_data = {season: ts[ts.index.month == i+1] for i, season in enumerate(seasons)}
        f_value, p_value = friedmanchisquare(*seasons_data.values())
        if p_value <= 0.05:
            print(f"El p-valor es {p_value}, por lo tanto, se puede inferir que hay patrones estacionales significativos.")
        else:
            print(f"El p-valor es {p_value}, por lo tanto, no se puede inferir que hay patrones estacionales significativos.")
        return f_value, p_value
    @staticmethod
    def PeriodogramTest(data):
        """Función periodogram_test

        Realiza una prueba de periodograma para determinar la presencia de periodicidad o estacionalidad en una serie de tiempo dada.

        Parámetros:
        - data: un array o una lista de valores numéricos que representan la serie de tiempo.

        Retorno:
        - una tupla con dos elementos:
        * max_freq: la frecuencia con el valor máximo de potencia en el periodograma.
        * inference: un string que indica si hay o no hay evidencia de periodicidad en los datos.
            
        """
        freq, power = periodogram(data)
        plt.plot(freq, power)
        plt.xlabel('Frequency')
        plt.ylabel('Power')
        plt.show()
        
        max_freq = freq[np.argmax(power)]
        alpha = 0.05
        threshold = 1 - alpha
        
        if max_freq < threshold:
            return (max_freq, 'No hay evidencia de periodicidad en los datos')
        else:
            return (max_freq, 'Hay evidencia de periodicidad en los datos')




    

class modeling:
    def __init__(self):
        """
        Class constructor
        """
        print("Class modeling...", self)
    
    def grid_search(model, param_dist:dict):
        '''
        This function search the best parameters with a parameter distributions dictionary with the parameters of the respective model using cross validation with randomizedSearchCV
        Args:
            model: model to be used for training
            param_dist: dictionary with the parameter distributions of the model
        Returns:
            grid_search_cv: trained model with the best parameters
        '''
        grid_search_cv= RandomizedSearchCV(model,
                                        param_distributions = param_dist,
                                        cv = 5,
                                        n_iter = 10,
                                        n_jobs = None,
                                        random_state= 0)
        return grid_search_cv

    def roc_curve_graph(fpr:np.ndarray, tpr:np.ndarray):
            '''
            This function creates a ROC curve for the forecast of interest of a binary variable
            Args:
                fpr: false positive rate
                tpr: true positive rate
            '''
            plt.figure(1)
            plt.plot(fpr, tpr, color="orange", label="ROC")
            plt.plot([0, 1], [0, 1], color="pink", linestyle="--")
            plt.plot(fpr, tpr, label='ROC Curve')
            plt.xlabel('Tasa falsos positivos')
            plt.ylabel('Tasa verdaderos positivos')
            plt.title('ROC curve')
            plt.legend(loc='best')
            plt.show()

    # Codification inputs with OneHotEncoder
    def categorical_transformer(category_columns:list, x:pd.DataFrame,):
        '''
        This function performs transformations for categorical variables using the OneHotEncoder preprocessor
        Args:
            category_columns: list with the names of the columns that contain categorical data
            x: original dataframe with the data to be transformed
        Returns:
            X: transformed dataframe with the categorical data encoded with OneHotEncoder
        '''
        category_columns=category_columns
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        preprocessor = ColumnTransformer(
                            transformers=[('cat', categorical_transformer, category_columns)],
                            remainder='passthrough'
                        )
        # Generar datos preprocesados
        df_preprocess = preprocessor.fit_transform(x)
        # Crear dataframe de datos preprocesados
        encoded_category = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(category_columns)
        labels=encoded_category
        transformation=preprocessor.transform(x)
        # print(labels.shape)
        # print(transformation.shape)
        X= pd.DataFrame(transformation, columns=labels)
        return X

    def categorical_transformer_pipeline(df_model: pd.DataFrame, dependiente:str, category_columns:list):
        '''
        This function encodes categorical data in a dataframe using the OneHotEncoder preprocessor within a ColumnTransformer object
        Args:
            df_model: dataframe with the data to be transformed
            dependiente: name of the dependent variable column
            category_columns: list with the names of the columns that contain categorical data
        Returns:
            X_train: transformed training data with the categorical data encoded with OneHotEncoder
            X_test: transformed testing data with the categorical data encoded with OneHotEncoder
            y_train: training data for the dependent variable
            y_test: testing data for the dependent variable
        '''

        X_train, X_test, y_train, y_test = train_test_split(df_model.drop([dependiente], axis='columns'),
                                                            df_model[dependiente],train_size=0.8,random_state=1212,shuffle=True)
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])
        preprocessor = ColumnTransformer(
                                                transformers=[('cat', categorical_transformer, category_columns)],
                                                remainder='passthrough'
                                            )
        # Generar datos preprocesados
        df_preprocess = preprocessor.fit_transform(X_train)

        # Crear dataframe de datos preprocesados
        encoded_category = preprocessor.named_transformers_['cat']['onehot'].get_feature_names(category_columns)
        labels=encoded_category
        print(labels.shape)
        print(preprocessor.transform(X_train).shape)
        # labels
        X_train= pd.DataFrame(preprocessor.transform(X_train), columns=labels)
        X_test= pd.DataFrame(preprocessor.transform( X_test), columns=labels)
        return X_train, X_test, y_train, y_test



    # Codification inputs
    def prepare_inputs(X_train, X_test):
        """
            Encode the categorical features of the input data using pd.get_dummies().

            Parameters:
            -----------
            X_train : pandas.DataFrame
                Training data to be encoded.
            X_test : pandas.DataFrame
                Testing data to be encoded.

            Returns:
            --------
            X_train_enc : pandas.DataFrame
                Encoded training data.
            X_test_enc : pandas.DataFrame
                Encoded testing data.
            """
        #categorical_features = ['canal_atencion','categoria','causa','medio','mercado', 'tipo_servicio']
        X_train_enc = pd.get_dummies(X_train, columns=X_train.columns, drop_first=False)
        X_test_enc  = pd.get_dummies(X_test, columns=X_test.columns, drop_first=False)
        return X_train_enc, X_test_enc

    def prepare_input_entire_sample(X):
        '''
        This function encode the covariates of df using pd.get_dummies with a previous list
        '''
        X_enc = pd.get_dummies(X, columns=X.columns, drop_first=False)
        return X_enc


    def grid_search(model, param_dist:dict):
        '''
        This function performs a randomized search over a specified parameter distribution dictionary
        to find the best hyperparameters for the given model.

        Parameters:
        model: The machine learning model object on which hyperparameter tuning will be performed.
        param_dist (dict): The parameter distribution dictionary containing the hyperparameters for the model.

        Returns:
        grid_search_cv: The RandomizedSearchCV object with the specified hyperparameters.
        '''
        grid_search_cv= RandomizedSearchCV(model,
                                        param_distributions = param_dist,
                                        cv = 5,
                                        n_iter = 5,
                                        n_jobs = None,
                                        random_state= 0)
        return grid_search_cv

    def roc_curve_graph(fpr:np.ndarray, tpr:np.ndarray):
        '''
        This function creates a Roc Curve for the forescast of interest of a binary variable.
        '''
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.xlabel('Tasa falsos positivos')
        plt.ylabel('Tasa verdaderos positivos')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()

    def plot_data(df):
        for col in df.columns:
            plt.figure(figsize=(12,4))
            plt.plot(df[col])
            plt.title(f"Serie de tiempo para '{col}'")
            plt.xlabel("Fecha")
            plt.ylabel("Valor")
            plt.grid(True)
            plt.show()


    def apply_differencing(dataframe, columns):
        df_diff = dataframe.copy()  # Crear una copia del DataFrame original

        for column in columns:
            values = dataframe[column].values  # Obtener los valores de la columna

            diff_values = np.diff(values)  # Aplicar diferenciación a los valores

            # Asignar la columna diferenciada al nuevo DataFrame
            df_diff[column] = np.concatenate([[np.nan], diff_values])
        return df_diff

    # def apply_differencing(dataframe, columns):
    #     df_diff = dataframe.copy()  # Crear una copia del DataFrame original
        
    #     for column in columns:
    #         values = dataframe[column].values  # Obtener los valores de la columna
            
    #         diff_values = np.diff(values)  # Aplicar diferenciación a los valores
            
    #         # Agregar el valor NaN al inicio de la columna diferenciada
    #         #diff_column = np.insert(diff_values, 0, np.nan)
            
    #         # Asignar la columna diferenciada al nuevo DataFrame
    #         df_diff[column] = diff_values
        
    #     return df_diff

    
    def boxcox_transform(df):
        """
        Aplica la transformación de Box-Cox a todas las columnas de un DataFrame que contengan datos numéricos.
        Si el valor de lambda es cercano a 1, no se aplica ninguna transformación. Si lambda tiende a 0, se recomienda
        aplicar la transformación logarítmica. Si lambda es cercano a 0.5, se recomienda aplicar la transformación de
        raíz cuadrada.

        Parameters
        ----------
        df : pandas.DataFrame
            El DataFrame que contiene los datos que se van a transformar.

        Returns
        -------
        transformed_df : pandas.DataFrame
            Un DataFrame con las mismas columnas que el DataFrame de entrada, pero con los datos transformados.

        Raises
        ------
        ValueError
            Si el DataFrame no contiene columnas numéricas.
        """
            
        transformed_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            data = df[col].dropna()
            if data.min() <= 0: #si el valor mínimo de los datos es menor o igual a cero, se realiza una corrección para que todos los valores sean positivos.
                data = data + abs(data.min()) + 0.1
            transformed_data, lam = stats.boxcox(data)
            _, p = stats.normaltest(transformed_data)
            if abs(1 - lam) < 0.1:
                transformed_df[col] = data
                print(f'Columna {col}: lambda={lam:.2f}. No se necesita transformación.')
            elif abs(0.5 - lam) < 0.1:
                transformed_df[col] = np.sqrt(data)
                print(f'Columna {col}: lambda={lam:.2f}. Se recomienda la transformación de raíz cuadrada.')
            elif lam < 0.1:
                transformed_df[col] = np.log(data)
                print(f'Columna {col}: lambda={lam:.2f}. Se recomienda la transformación logarítmica.')
            else:
                transformed_df[col] = transformed_data
                print(f'Columna {col}: lambda={lam:.2f}. Transformada con éxito.')
            if p < 0.05:
                print(f'    El p-valor para la columna {col} es {p:.4f}, por lo que se sugiere verificar la distribución transformada.')
            else:
                print(f'    El p-valor para la columna {col} es {p:.4f}, la distribución transformada parece ser normal.')
        return transformed_df
    
    def yeo_johnson_test(df):
        """
        Transforma cada columna del dataframe utilizando la transformación de Yeo-Johnson
        si se cumple alguno de los siguientes criterios:
        - El valor de lambda es menor a 0.5 o mayor a 1.5.
        - El p-valor de la prueba de normalidad para los datos transformados es menor a 0.05.

        Parameters
        ----------
        df : pandas.DataFrame
            El dataframe a transformar.

        Returns
        -------
        transformed_df : pandas.DataFrame
            El dataframe transformado.

        """
        transformed_df = pd.DataFrame(index=df.index)
        for col in df.columns:
            data = df[col].dropna()
            transformed_data, lam = stats.yeojohnson(data)
            _, p = stats.normaltest(transformed_data)
            if abs(lam - 1) > 0.5 or p < 0.05:
                transformed_df[col] = transformed_data
                if abs(lam - 1) > 0.5 and lam < 1:
                    print(f'Columna {col}: lambda={lam:.2f}. Se recomienda la transformación logarítmica.')
                elif abs(lam - 1) > 0.5 and lam > 1:
                    print(f'Columna {col}: lambda={lam:.2f}. Se recomienda la transformación inversa.')
                else:
                    print(f'Columna {col}: lambda={lam:.2f}. Transformada con éxito.')
            else:
                transformed_df[col] = data
                print(f'Columna {col}: lambda={lam:.2f}. No se necesita transformación.')
        return transformed_df

    # Si lambda es igual a 0, aplicar la transformación logarítmica a los datos.
    # Si lambda es diferente de 0, aplicar la transformación Yeo-Johnson a los datos.
    # Si el valor absoluto de lambda es menor que 0.5, se considera una transformación cercana a cero y se sugiere aplicar una transformación adicional, como la raíz cuadrada.
    # Si el valor absoluto de lambda es mayor o igual a 0.5 y menor que 1, se considera una transformación moderada y no se sugiere ninguna transformación adicional.
    # Si el valor absoluto de lambda es mayor o igual a 1, se considera una transformación fuerte y no se sugiere ninguna transformación adicional.

    def plot_comparison(original_df, transformed_df):
        """
        Esta función toma dos dataframes (uno con los datos originales y otro con los datos transformados) y grafica cada columna de ambos dataframes en la misma figura.

        Parámetros:
        - original_df: Dataframe con los datos originales.
        - transformed_df: Dataframe con los datos transformados.
        """
        for col in original_df.columns:
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))
            fig.suptitle(f'Comparación de {col}', fontsize=16)

            # Gráfico de la serie original
            ax[0].plot(original_df[col], label='Original')
            ax[0].set_title('Original')
            ax[0].legend()

            # Gráfico de la serie transformada
            ax[1].plot(transformed_df[col], label='Transformada')
            ax[1].set_title('Transformada')
            ax[1].legend()

            plt.show()

    def optimize_SARIMA(y, x, d, D, s, parametros=[3, 3, 3, 3], validacion=12, forecasting_steps=12):
        """
        Esta función calcula múltiples modelos SARIMA y retorna un listado de los parámetros del modelo 
        con mejor desempeño predictivo fuera de muestra en términos del Error Porcentual Absoluto Medio (MAPE). 

        PARÁMETROS:
        y: Variable Endógena
        d: Orden de Integración Regular
        D: Orden de Integración Estacional
        s: Periodicidad o duración de la temporada
        parametros: máximo orden del valor de los parámetros (p, q, P, Q)
        validacion: número de registros para validar el modelo.
        """

        # Cut data: Training and Testing
        y_train = y[:len(y) - validacion]
        y_test = y[len(y) - validacion:]
        x_train = x[:len(y) - validacion]
        x_test = x[len(y) - validacion:]

        # Parameters
        p = range(0, parametros[0] + 1, 1)
        q = range(0, parametros[1] + 1, 1)
        P = range(0, parametros[2] + 1, 1)
        Q = range(0, parametros[3] + 1, 1)
        parameters = product(p, q, P, Q)
        parameters_list = list(parameters)

        results = []

        # Estimate
        for param in tqdm_notebook(parameters_list):
            try:
                model = SARIMAX(y_train, exog=x_train, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], s)).fit(disp=-1)
            except:
                continue

            y_pred = model.forecast(y_test.shape[0], exog=x_test)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100
            rmse = mean_squared_error(y_test, y_pred, squared=False)
            results.append([param, mape, rmse])

        result_df = pd.DataFrame(results)
        result_df.columns = ['(p,q)x(P,Q)', 'MAPE', 'RMSE']

        # Sort in ascending order, lower MAPE is better
        result_df = result_df.sort_values(by='MAPE', ascending=True).reset_index(drop=True).iloc[0, :]
        parameters = result_df[0]
        mape = result_df[1]
        rmse = result_df[2]

        # Estimate with all data
        best_model = SARIMAX(y, exog=x, order=(parameters[0], d, parameters[1]),
                            seasonal_order=(parameters[2], D, parameters[3], s)).fit(disp=-1)

        # Forecasting
        forecast = best_model.forecast(steps=forecasting_steps, exog=x_test)

        return {'forecast': forecast, 'parameters': parameters, 'mape': mape, 'rmse': rmse}
            



class exploratory_tools:
    
    def __init__(self):
        print("Class exploratory_tools...", self)

     # histogramas: variables categóricas
    #......................................

    def count_graph(data: pd.DataFrame, covariate: list, response: str): 
        '''
        This function creates multiple countplots of a DataFrame relative to
        a variable of interest of binary nature.

        Parameters:
        data (pd.DataFrame): DataFrame to plot.
        covariate (list): List of covariates to plot against response.
        response (str): Response variable to plot against.

        Returns:
        None. The function plots multiple countplots.

        '''
        for k in covariate:
            f, ax = plt.subplots(figsize=(12, 3))
            sns.countplot(data=data.dropna(subset=['{0}'.format(k)]), 
                        x="{0}".format(k), hue=response, palette=['#432371',"#FAAE7B"]).set(title='Frecuencia de: {0}'.format(k))
            plt.show()

    def density_graph(data: pd.DataFrame, col: list):
        '''
        This function creates multiple density plots of a dataframe relative to a binary variable of interest.

        Parameters:
        -----------
        data: pandas DataFrame
            The input dataframe containing the variables of interest.

        col: list
            A list containing the column names for which density plots will be generated.

        Returns:
        --------
        None
        '''

        for i in col:
            plt.figure(figsize=(10,4))
            ax = sns.kdeplot(df[i][df.imputables == 1 ], color="darkturquoise", shade=True)
            sns.kdeplot(df[i][df.imputables == 0 ], color="lightcoral", shade=True)
            plt.legend(['Imputable', 'No Imputable'])
            plt.title('Density Plot')
            ax.set(xlabel='estrato')
            plt.xlim(-2,8)
            plt.show()

    # Variable Response Graph
    #......................................
    def response_graph(df:pd.DataFrame, res:str):
            '''
            Crea múltiples gráficos de barras de la variable de interés que es binaria.

            Parameters:
            -----------
            df : pandas.DataFrame
                El DataFrame que contiene los datos.
            res : str
                El nombre de la columna que contiene la variable de interés.

            Returns:
            --------
            None
            '''
            sns.countplot(data=df, x=res, palette='flare').set(title='Diagrama de Barras de la Variable de Interés')
            plt.show()
            print(df[res].value_counts()/df.shape[0])
            print('La mayoria:  is %s.' %df[res].value_counts().idxmax())

    #Función de asociación entre variables categoricas
    def contingency_table(variable1,variable2):
            '''
            Computes a contingency table and performs a chi-squared test to determine the degree of association
            between two categorical variables.

            Parameters:
            variable1 (pandas.Series): First categorical variable.
            variable2 (pandas.Series): Second categorical variable.

            Returns:
            None. Prints the result of the chi-squared test.
            '''
            tabla=pd.crosstab(index=variable1,columns=variable2, margins=True)
            obs=np.array([np.array(tabla)])
            stat, p, dof, expected = chi2_contingency(obs)
            prob = 0.95
            critical = chi2.ppf(prob, dof)
            if abs(stat) >= critical:
                print('Dependent (reject H0)')
            else:
                print('Independent (fail to reject H0): Acept')
    
    #Función para gráficar los porcentajes de categoria con respecto a la variable de interés
    def plotly_percentage(covariable, dependiente,df):
        '''
        Generates a stacked bar chart using Plotly to show the relative frequencies of a dependent variable for each level of a given independent variable
        
        Parameters:
        covariable (str): The name of the independent variable
        dependiente (str): The name of the dependent variable
        df (pd.DataFrame): The pandas DataFrame containing the data to be plotted
        
        Returns:
        None
        '''
        group=df.groupby([covariable,dependiente]).agg({covariable : 'count'})
        group=group.rename(columns={covariable:'count'})
        group=group.reset_index()
        group['percentage']=(group['count']/group['count'].sum())*100
        fig = px.bar(group.assign(bar=6),
                x=covariable,
                y="percentage",
                color=dependiente, 
                color_discrete_map={0: 'lightcyan', 1: 'royalblue'},
                barmode='stack',
                text=group['percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
                title='Frecuencias relativas de {0}'.format(covariable)).update_layout(
                xaxis_visible=True, autosize=True, width=800
        )
        fig.update_traces(marker_color= ['#008080','#7fbfbf']*len(group))
        fig.show()


    # Función para ver nulos

    def null_proportion(data: pd.DataFrame)->pd.DataFrame:
        '''
        This function calculates the number of unique, null values ​​and their relative weight.
        Parameters
        ----------
        data : pd.DataFrame
        Returns
        -------
        nulos : pd.DataFrame
            DataFrame with the number of missing values ​​and their relative weight with respect to the total data.
        '''
        
        df = data.copy()
        nulos = pd.DataFrame()
        total = df.isnull().sum().sort_values(ascending=False)
        percent = ((df.isnull().sum() / df.isnull().count()) * 100).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        missing_data = missing_data.reset_index()
        missing_data.columns = ['Name', 'Total', 'Percent']
        return missing_data

    def category_impute(cols_to_impute:list,df:pd.DataFrame):
        for i in cols_to_impute:
         df[i].fillna(df[i].mode().iloc[0],inplace=True)
         return print(df.isnull().sum())
        

    def fill_null_values_with_rolling_mean(df, column_names, window_size):
        for col in column_names:
            for i in range(len(df[col])):
                if pd.isnull(df[col].iloc[i]):
                    if i >= window_size:
                        df[col].iloc[i] = df[col].iloc[i - window_size:i].mean()
                    else:
                        df[col].iloc[i] = df[col].iloc[:i].mean()
        return df
    
    def reemplazar_guion(fecha):
        if '-' in fecha:
            return fecha.replace('-', '/')
        else:
            return fecha
        



import os
import pandas as pd







class ParquetFilterResampler:
    def filtrar_exportar_parquet(self, ruta, fecha_inicio, fecha_fin, nombre_archivo, startwith):
        """
        Filtra los archivos .data en la ruta especificada por fecha y exporta el dataframe resultante en formato parquet.

        Parámetros
        ----------
        ruta : str
            Ruta donde se encuentran los archivos .data.
        fecha_inicio : str
            Fecha de inicio del filtro en formato 'YYYY-MM-DD'.
        fecha_fin : str
            Fecha de fin del filtro en formato 'YYYY-MM-DD'.
        nombre_archivo : str
            Nombre del archivo parquet resultante sin la extensión.

        Retorna
        -------
        None
            Exporta el dataframe resultante en formato parquet en la misma ruta especificada.
        """
        # Obtener lista de archivos .data en la ruta dada
        archivos = [archivo for archivo in os.listdir(ruta) if archivo.endswith('.data') and archivo.startswith(startwith)]

        # Crear una lista vacía para almacenar los dataframes de cada archivo
        lista_df = []

        # Iterar sobre los archivos y leerlos como dataframes de pandas
        for archivo in archivos:
            ruta_archivo = os.path.join(ruta, archivo)
            df = pd.read_csv(ruta_archivo, delimiter="|", parse_dates=["Fecha"])
            # Filtrar por fecha
            df = df[(df["Fecha"] >= fecha_inicio) & (df["Fecha"] <= fecha_fin)]
            lista_df.append(df)

        # Concatenar los dataframes por filas
        df_final = pd.concat(lista_df, axis=0)

        # Exportar dataframe resultante en formato parquet
        ruta_archivo_parquet = os.path.join(ruta, f"{nombre_archivo}.parquet")
        df_final.to_parquet(ruta_archivo_parquet)
        

    

class rnn_tools:
    
    def __init__(self):
        print("Class Red Neuronal Recurrente tools.......", self)


    def create_simple_rnn(hp):
        """
        Crea un modelo de red neuronal con capas SimpleRNN configurables mediante Hyperparameter Tuning.
        
        Parameters:
            hp (keras_tuner.Hyperparameters): Objeto Hyperparameters para configurar la búsqueda de hiperparámetros.
        Returns:
            tf.keras.models.Sequential: El modelo de red neuronal configurado con las capas SimpleRNN y otros hiperparámetros definidos.
        """
        model = Sequential() # Modelo secuencial: pila lineal de capas.

        # Capa SimpleRNN inicial
        model.add(SimpleRNN(units=hp.Int(name="rnn_units", min_value=32, max_value=128, step=32),
                            activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]),
                            return_sequences=True,
                            input_shape=(1, 49))) # Forma de los datos de entrada: una sola secuencia de longitud 1 con 17 características.

        # Capa de Dropout inicial
        model.add(Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.7, step=0.1)))
        
        # Agregar capas SimpleRNN dinámicamente
        n_layers = hp.Int(name="n_layers", min_value=1, max_value=4)
        for i in range(n_layers):
            model.add(SimpleRNN(units=hp.Int(name=f"rnn_{i}_units", min_value=32, max_value=128, step=32),
                                activation=hp.Choice(name=f"activation_{i}", values=["relu", "selu", "sigmoid", "tanh"]),
                                return_sequences=True))
        
        # Capa SimpleRNN final fuera del bucle
        model.add(SimpleRNN(units=hp.Int(name=f"rnn_final_units", min_value=32, max_value=128, step=32),
                            activation=hp.Choice(name="activation_final", values=["relu", "selu", "sigmoid", "tanh"])))
        
        # Capa de Dropout final
        model.add(Dropout(rate=hp.Float(name="dropout_rate_final", min_value=0.2, max_value=0.7, step=0.1)))
        
        # Capa Dense de salida
        model.add(Dense(units=1, activation="linear")) # tiene una sola unidad neuronal, típicamente utilizado en problemas de regresión
        
        # Compilar el modelo
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
        
        return model
    


    def create_lstm(hp):
        """
        Crea un modelo de red neuronal con capas LSTM configurables mediante Hyperparameter Tuning.
        
        Parameters:
            hp (keras_tuner.Hyperparameters): Objeto Hyperparameters para configurar la búsqueda de hiperparámetros.
        Returns:
            tf.keras.models.Sequential: El modelo de red neuronal configurado con las capas LSTM y otros hiperparámetros definidos.
        """
        model = Sequential()
        model.add(layer=LSTM(units=hp.Int(name="input_unit", min_value=32, max_value=128, step=32),
                            activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]),
                            return_sequences=True, input_shape=(1, 49)))
        model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
        for i in range(hp.Int(name="n_layers", min_value=1, max_value=4)):
            model.add(layer=LSTM(units=hp.Int(name=f"lstm_{i}_units", min_value=32, max_value=128, step=32),
                                activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]),
                                return_sequences=True))
        model.add(layer=LSTM(units=hp.Int(name=f"lstm_{i}_units", min_value=32, max_value=128, step=32),
                            activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"])))
        model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
        model.add(layer=Dense(units=1, activation="linear"))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
        return model

    
    def create_gru(hp):
        """
        Crea un modelo de red neuronal con capas GRU configurables mediante Hyperparameter Tuning.
        
        Parameters:
            hp (keras_tuner.Hyperparameters): Objeto Hyperparameters para configurar la búsqueda de hiperparámetros.
        Returns:
            tf.keras.models.Sequential: El modelo de red neuronal configurado con las capas GRU y otros hiperparámetros definidos.
        """
        model = Sequential()
        model.add(layer=GRU(units=hp.Int(name="input_unit", min_value=32, max_value=128, step=32),
                            activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]),
                            return_sequences=True, input_shape=(1, 49)))
        model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
        for i in range(hp.Int(name="n_layers", min_value=1, max_value=4)):
            model.add(layer=GRU(units=hp.Int(name=f"gru_{i}_units", min_value=32, max_value=128, step=32),
                                activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]),
                                return_sequences=True))
        model.add(layer=GRU(units=hp.Int(name=f"gru_{i}_units", min_value=32, max_value=128, step=32),
                            activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"])))
        model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
        model.add(layer=Dense(units=1, activation="linear"))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics="mse")
        return model

    def create_bidirectionallstm(hp):
        """
        Crea un modelo de red neuronal con capas Bidirectional LSTM configurables mediante Hyperparameter Tuning.
        
        Parameters:
            hp (keras_tuner.Hyperparameters): Objeto Hyperparameters para configurar la búsqueda de hiperparámetros.
        Returns:
            tf.keras.models.Sequential: El modelo de red neuronal configurado con las capas Bidirectional LSTM y otros hiperparámetros definidos.
        """
                
        model = Sequential()
        model.add(layer=Bidirectional(LSTM(units=hp.Int(name="input_unit", min_value=32, max_value=128, step=32),
                                        activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]),
                                        return_sequences=True, input_shape=(1, 49)))) 
        model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
        for i in range(hp.Int(name="n_layers", min_value=1, max_value=4)):
            model.add(layer=Bidirectional(LSTM(units=hp.Int(name=f"bilstm_{i}_units", min_value=32, max_value=128, step=32),
                                            activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]),
                                            return_sequences=True)))
        model.add(layer=Bidirectional(LSTM(units=hp.Int(name=f"bilstm_{i}_units", min_value=32, max_value=128, step=32),
                                        activation=hp.Choice(name="activation", values=["relu", "selu", "sigmoid", "tanh"]))))
        model.add(layer=Dropout(rate=hp.Float(name="dropout_rate", min_value=0.2, max_value=0.5, step=0.1)))
        model.add(layer=Dense(units=1, activation="linear"))
        model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mse"])
        return model
    


class RegressionEvaluator(object):
    def __init__(self, predicted: pd.Series, observed: pd.Series):
        self.predicted = predicted
        self.observed = observed
        self.metrics = None

    def calculate_metrics(self):
        self.metrics = {"rmse": np.round(mean_squared_error(y_true=self.observed,
                                                            y_pred=self.predicted,
                                                            squared=False), decimals=4),
                        "mae": np.round(mean_absolute_error(y_true=self.observed,
                                                            y_pred=self.predicted), decimals=4),
                        "mape": np.round(mean_absolute_percentage_error(y_true=self.observed,
                                                                        y_pred=self.predicted), decimals=4)}
        return self.metrics

    def print_metrics(self):
        if self.metrics is None:
            self.calculate_metrics()
        print(f"El RMSE es: {self.metrics['rmse']}")
        print(f"El MAE es: {self.metrics['mae']}")
        print(f"El MAPE es: {self.metrics['mape']}")

class Prediction_intervals_tools:
    
    def __init__(self):
        print("Class Prediction Intervals tools.......", self)

    # Loss function for each quantile (pinball_loss)
    # ==============================================================================
    def mean_pinball_loss_q10(y_true, y_pred):
        """
        Pinball loss for quantile 10.
        """
        return mean_pinball_loss(y_true, y_pred, alpha=0.1)


    def mean_pinball_loss_q90(y_true, y_pred):
        """
        Pinball loss for quantile 90.
        """
        return mean_pinball_loss(y_true, y_pred, alpha=0.9)
