import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def ADX(df):
    n_adx = 14  # El período para el cálculo del ADX
    n = 14

    # Calcular el True Range (TR)
    df['High-Low'] = df['High'] - df['Low']
    df['High-Close-Prev'] = abs(df['High'] - df['Close'].shift(1))
    df['Low-Close-Prev'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['High-Low', 'High-Close-Prev', 'Low-Close-Prev']].max(axis=1)

    # Calcular el Indicador Direccional Positivo (+DI) y el Indicador Direccional Negativo (-DI)
    df['High-Prev'] = df['High'].shift(1)
    df['Low-Prev'] = df['Low'].shift(1)

    df['+DM'] = 0.0
    df['-DM'] = 0.0

    df.loc[(df['High'] - df['High-Prev'] > df['Low-Prev'] - df['Low']), '+DM'] = df['High'] - df['High-Prev']
    df.loc[(df['Low-Prev'] - df['Low'] > df['High'] - df['High-Prev']), '-DM'] = df['Low-Prev'] - df['Low']

    df['+DI'] = (df['+DM'].rolling(window=n, min_periods=1).sum() / df['TR'].rolling(window=n, min_periods=1).sum()) * 100
    df['-DI'] = (df['-DM'].rolling(window=n, min_periods=1).sum() / df['TR'].rolling(window=n, min_periods=1).sum()) * 100

    # Calcular el Índice de Movimiento Direccional Promedio (ADX)
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])) * 100
    df['ADX'] = df['DX'].rolling(window=n_adx, min_periods=1).mean()

    # Eliminar columnas intermedias si es necesario
    ADX_results = np.array(df[['ADX']].values.tolist()).squeeze()
    df.drop(['High-Low', 'High-Close-Prev', 'Low-Close-Prev', 'TR', 'High-Prev', 'Low-Prev', '+DM', '-DM', '+DI', '-DI', 'DX', 'ADX'], axis=1, inplace=True)

    return ADX_results

class ETL: # Extract, transform and load data
    def __init__(self, target) -> None:
        self.symbols = ['SAB.MC', 'SAN.MC', 'BKT.MC', 'BBVA.MC', 'CABK.MC', 'UNI.MC', '^IBEX', '^IXIC','^DJI', '^GSPC','^N225', '^STOXX50E', 'EURUSD=X', 'EURGBP=X','EURJPY=X'] # Los 6 bancos del IBEX 35 y otros activos relacionados
        self.assets_used = []
        if target in self.symbols:
            self.df = yf.Ticker(target).history(period='max').reset_index()[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']] # Pandas df
            self.df['Date'] = [datetime.strptime(a.strftime('%Y-%m-%d'),'%Y-%m-%d') for a in self.df['Date']] # Cambio formato de fechas a AAAA-MM-DD 
        else: raise Exception('Crea una instancia con el símbolo de un banco del IBEX 35')

        self.tech_df = self.technical_indicators()
        self.fft_df = self.fourier_transform()
        self.assets_df = self.related_assets(target)

        self.df = self.df[-len(self.tech_df.index):].reset_index().drop(['index'], axis=1)    # Ajusto los 3 dataframes al mismo número de ejemplos (5000 de momento)
        self.fft_df = self.fft_df[-len(self.tech_df.index):].reset_index().drop(['index'], axis=1)   # ya que los primeros de tech_df eran NaN y fueron eliminados
        self.dates = np.array(self.df['Date']).reshape(-1,1)[-len(self.tech_df.index):]

        self.data_tensor, self.scaled_data_tensor, self.scaled_train_tensor, self.scaled_val_tensor = self.preprocess() # A priori solo voy a usar scaled_data_tensor en el VAE

    def related_assets(self, target): # Obtenemos df de empresas relacionadas
        assets = []
        for symb in self.symbols:
            if symb != target:
                obj = yf.Ticker(symb).history(period='max').reset_index()[['Date', 'Close']]
                obj['Date'] = [datetime.strptime(a.strftime('%Y-%m-%d'),'%Y-%m-%d') for a in obj['Date']]
                if len(obj.index) > len(self.tech_df.index): 
                    assets.append(obj[-len(self.tech_df.index):].reset_index().drop(['index'], axis=1))
                    self.assets_used.append(symb)
                # Los assets con menos datos no los voy a utilizar
        return assets

    def technical_indicators(self): # Calculamos los indicadores técnicos de nuestro banco
        tech_df = self.df[['Date']].sort_values(by='Date') # Ordena el DataFrame por fecha si no está ordenado
        
        # Cálculo de la Media Móvil Simple (SMA) de 7 y 21 días
        tech_df['SMA_7'] = self.df['Close'].rolling(window=7).mean()
        tech_df['SMA_21'] = self.df['Close'].rolling(window=21).mean()

        # Cálculo de la Media Móvil Exponencial (EMA) = EWMA con periodo de 10 días (corto plazo)
        tech_df['EMA'] =self.df['Close'].ewm(span=10).mean()

        # Cálculo de las bandas de Bollinger
        tech_df['sigma_20'] = self.df['Close'].rolling(window=20).std() # Desviación estándar en periodo de 20 días
        tech_df['uBB'] = tech_df['SMA_21'] + 2 * tech_df['sigma_20'] # Banda de Bollinger superior
        tech_df['lBB'] = tech_df['SMA_21'] - 2 * tech_df['sigma_20'] # Banda de Bollinger superior

        # Cálculo de la Media Móvil de Convergencia/Divergencia (MACD) (por definición EMA26-EMA12)
        ema_short = self.df['Close'].ewm(span=12, adjust=False).mean() # EMA a corto plazo
        ema_long = self.df['Close'].ewm(span=26, adjust=False).mean() # EMA a largo plazo
        tech_df['MACD'] = ema_short - ema_long
        tech_df['SignalLine'] = tech_df['MACD'].ewm(span=9, adjust=False).mean() # Línea de señal a 9 días (definición)

        # Cálculo de Índice de fuerza relativa (RSI) en 14 días (https://es.wikipedia.org/wiki/%C3%8Dndice_de_fuerza_relativa)
        tech_df['Diff'] = self.df['Close'].diff(1) # Diferencias entre precios consecutivos
        
        tech_df['Gain'] = np.where(tech_df['Diff'] > 0, tech_df['Diff'], 0) # Valores positivos (ganancias) y negativos (pérdidas)
        tech_df['Lose'] = np.where(tech_df['Diff'] < 0, -tech_df['Diff'], 0) # np.where(condicion, si se cumple, si no)

        tech_df['EMA_Gain'] = tech_df['Gain'].rolling(window=14).mean() # (EMA) de las ganancias y pérdidas
        tech_df['EMA_Lose'] = tech_df['Lose'].rolling(window=14).mean() 
 
        tech_df['RS'] = tech_df['EMA_Gain'] / tech_df['EMA_Lose'] # Calculo del RSI
        tech_df['RSI'] = 100 - (100 / (1 + tech_df['RS'])) 

        tech_df = tech_df.drop(['Diff', 'Gain', 'Lose', 'EMA_Gain', 'EMA_Lose', 'RS'], axis=1) # Eliminamos columnas innecesarias

        # Cálculo del Oscilador Estocástico --> compara actual con máx y mín de un periodo dado (14 días)
        tech_df['min'] = self.df['Close'].rolling(window=14).min()
        tech_df['max'] = self.df['Close'].rolling(window=14).max()
        tech_df['%K'] = 100 * ((self.df['Close'] - tech_df['min']) / (tech_df['max'] - tech_df['min']))
        tech_df['%D'] = tech_df['%K'].rolling(window=3).mean() # SMA DE %K
        tech_df = tech_df.drop(['min', 'max'], axis=1) # Eliminamos columnas innecesarias 

        # cálculo de ADX (más laborioso con función externa). Periodo de 14 días.
        tech_df['ADX'] = ADX(self.df)

        return tech_df[-5000:].reset_index().drop(['index'],axis=1)    # Voy a quedarme con los últimos 5000 datos para tener assets relacionados con esa cantidad de datos
    
    def fourier_transform(self): # Cálculo de transformadas de Fourier (mediante FFT)
        # Vamos a calcularla con distinto número de componentes para extraer tendencias de larga y corta distancia
        fft_values = np.fft.fft(self.df['Close'].values)
        fft_df = self.df[['Date']].sort_values(by='Date')

        for comp in [3, 6, 9,25, 100]:
            fft_freqs= np.copy(fft_values)
            fft_freqs[comp:-comp]=0 # Me quedo solo con el número 'comp' de frecuencias principales eliminando las intermedias
            fft_df['ifft_'+str(comp)] = np.fft.ifft(fft_freqs) # Transformada inversa sobre las frecuencias que he salvado
        
        return fft_df
    
    def preprocess(self):
        lista_dfs = [self.df[['Close', 'Open', 'High', 'Low', 'Volume']], self.tech_df, self.fft_df.apply(np.real)] + self.assets_df
        self.combined_df = pd.concat(lista_dfs, axis=1).drop(['Date'], axis=1) # Creamos un dataframe con todos los datos con los que contamos

        data_tensor = tf.convert_to_tensor(self.combined_df, dtype=tf.float32) # Generamos el tensor completo original para la siguiente fase del VAE
        train_data_tensor = data_tensor[:int(data_tensor.shape[0]*0.8)] # Este y el siguiente no los devuelvo pues no tienen uso
        val_data_tensor = data_tensor[int(data_tensor.shape[0]*0.8):]

        scaler = MinMaxScaler()  # Vamos a normalizar los datos con método min-max
        scaled_train_tensor = scaler.fit_transform(train_data_tensor) # Generamos tensor para entrenar el VAE
        scaled_val_tensor = scaler.transform(val_data_tensor) # Generamos tensor para evaluar el VAE
        scaled_data_tensor = scaler.transform(data_tensor) # Generamos tensor para usar el VAE entrenado con todo el dataset
        
        return data_tensor, scaled_data_tensor, scaled_train_tensor, scaled_val_tensor