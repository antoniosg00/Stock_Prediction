import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, LSTM, GRU, Input, BatchNormalization, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.regularizers import L1L2

from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.metrics import MeanAbsolutePercentageError as MAPE
from tensorflow.keras.metrics import MeanAbsoluteError as MAE
from tensorflow.keras.metrics import RootMeanSquaredError as RMSE

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

class StockPriceLSTM:
    """
    Este código implementa una red neuronal recurrente (LSTM) diseñada para predecir los precios futuros de acciones a partir de datos históricos. El objetivo es entrenar la LSTM para que aprenda a generar secuencias de precios de acciones futuras basándose en datos de entrada históricos.

    La red LSTM consta de una serie de capas LSTM que procesan los datos de entrada secuencialmente y producen una predicción de precio para el siguiente paso de tiempo.

    A medida que avanzan las épocas de entrenamiento, se espera que la LSTM sea capaz de producir predicciones precisas de los precios de las acciones para el día siguiente, lo que puede ser valioso en aplicaciones financieras y de inversión.
    
    """
    def __init__(self, n_features, timesteps, layers_type='LSTM', model_layer_dims=[128,64], momentum=0.9, dropout_rate=0.5, initial_lr=0.01, decay_steps=100, decay_rate=1, kernel_initializer='glorot_uniform', l1_factor=0.01, l2_factor=0.01, bidirectional=False):
        """
        Inicializa la clase StockPriceLSTM con parámetros y modelos.
        
        Args:
        n_features (int): Número de características en cada paso de tiempo.
        timesteps (int): Número de pasos de tiempo en los datos de entrada.
        layers_type (str): Arquitectura de capas LSTM o GRU
        model_layer_dims (list): Lista de dimensiones de las capas del modelo (por ejemplo, [128, 64]).
        momentum (float): Hiperparámetro de las capas BatchNormalization.
        dropout_rate (float): Tasa de dropout entre capas.
        initial_lr (float): Tasa de aprendizaje inicial.
        decay_steps (int): Pasos de decaimiento de la tasa de aprendizaje.
        decay_rate (float): Tasa de decaimiento de la tasa de aprendizaje.
        kernel_initializer (str): Inicializador de pesos para las capas (por defecto, 'glorot_uniform').
        l1_factor (float): Factor de regularización L1. 
        l2_factor (float): Factor de regularización L2.  
        bidirectional (bool): Bidireccionalidad de las capas LSTM o GRU.
        """
        
        # Establecemos las dimensiones adecuadas para los datos
        self.timesteps = timesteps
        self.n_features = n_features

        # Fijamos los hiperparámetros de las redes neuronales
        self.model_layer_dims = model_layer_dims 
        self.momentum = momentum # Hiperparámetro de las BatchNormalization
        self.dropout_rate = dropout_rate 
        self.kernel_initializer=kernel_initializer # O 'glorot_uniform' o 'he_normal'
        self.l1, self.l2 = l1_factor, l2_factor
        self.bidirectional = bidirectional

        # Fijamos los hiperparámetros de la tasa de aprendizaje
        lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate)
        self.model_optimizer = Adam(learning_rate=lr_schedule) # Adam con hiperparámetros producidos en la clase ExponentialDecay

        # Creamos el modelo
        if layers_type=='LSTM':
            self.model = self.build_lstm_model() # Modelo LSTM
        elif layers_type=='GRU':
            self.model = self.build_gru_model() # Modelo GRU
        else: raise Exception('ERROR AL CONSTRUIR EL MODELO: el atributo "layers_type" debe ser "LSTM" o "GRU".')

    def build_lstm_model(self):
        """
        Construye el modelo de la red LSTM.
        """
        lstm_model = Sequential()
        lstm_model.add(Input(shape=(self.timesteps, self.n_features)))  # Recordemos que keras ignora la primera dimensión (n_ejemplos) --> https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
        for i, units in enumerate(self.model_layer_dims): # Una o dos capas (no más para evitar overfitting) ¿O SÍ?
            if i==len(self.model_layer_dims)-1:
                if self.bidirectional: 
                    lstm_model.add(Bidirectional(LSTM(units, return_sequences=False, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2)))) # En la última capa return_sequences=False para obtener 1 solo valor
                else:
                    lstm_model.add(LSTM(units, return_sequences=False, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2)))
            else:
                if self.bidirectional:
                    lstm_model.add(Bidirectional(LSTM(units, return_sequences=True, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2)))) # Introduzco kernel_regularizer. ¿Qué es activity_regularizer (https://keras.io/api/layers/regularizers/)?
                else: 
                    lstm_model.add(LSTM(units, return_sequences=True, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2)))
            lstm_model.add(BatchNormalization(momentum=self.momentum))
            lstm_model.add(Dropout(self.dropout_rate))
        lstm_model.add(Dense(1, kernel_initializer=self.kernel_initializer)) # Salida lineal para predecir el precio del día siguiente

        lstm_model.compile(optimizer=self.model_optimizer, loss=MSE(), metrics=[MAPE(), MAE(), RMSE()]) # Se espera ver un MAPE alto pues los datos están normalizados en [0,1] (metrica de poca utilidad) 

        return lstm_model
    
    def build_gru_model(self):
        """
        Construye el modelo de la red GRU.
        """
        gru_model = Sequential()
        gru_model.add(Input(shape=(self.timesteps, self.n_features)))

        for i, units in enumerate(self.model_layer_dims):
            if i == len(self.model_layer_dims) - 1:
                if self.bidirectional:
                    gru_model.add(Bidirectional(GRU(units, return_sequences=False, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2))))
                else:
                    gru_model.add(GRU(units, return_sequences=False, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2)))
            else:
                if self.bidirectional:
                    gru_model.add(Bidirectional(GRU(units, return_sequences=True, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2))))
                else:
                    gru_model.add(GRU(units, return_sequences=True, activation='tanh', kernel_initializer=self.kernel_initializer, kernel_regularizer=L1L2(l1=self.l1, l2=self.l2)))
            gru_model.add(BatchNormalization(momentum=self.momentum))
            gru_model.add(Dropout(self.dropout_rate))

        gru_model.add(Dense(1, kernel_initializer=self.kernel_initializer))

        gru_model.compile(optimizer=self.model_optimizer, loss=MSE(), metrics=[MAPE(), MAE(), RMSE()])

        return gru_model
    
    def train(self, X_train, y_train, X_val, y_val, epochs, batch_size, path, patience=10):
        """
        Entrena la red en los datos de entrada.

        Args:
            X_train (numpy.ndarray): Datos de entrada con forma (n_ejemplos, timesteps, n_features).
            y_train (numpy.ndarray): Etiquetas correspondientes con forma (n_ejemplos, 1).
            X_val (numpy.ndarray): Datos de entrada con forma (n_ejemplos, timesteps, n_features).
            y_val (numpy.ndarray): Etiquetas correspondientes con forma (n_ejemplos, 1).
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del lote durante el entrenamiento.
        """
        # Añadimos un callback de EarlyStopping para interrumpir el entrenamiento cuando ya no mejore val_loss
        self.early_stopping = EarlyStopping(monitor='val_loss', patience=patience,  restore_best_weights=True, verbose=1)

        # Crear directorio para guardar checkpoints si no existe
        checkpoint_dir = path
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.h5')

        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor = 'val_loss',
            save_best_only=True,
            save_freq='epoch'
        )

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[checkpoint, self.early_stopping],
            verbose=1,
        )

        return history
    
    def train_opt(self, X_train, y_train, X_val, y_val, epochs, batch_size, patience=10):
        """
        Con esta función entrenamos el modelo en cada trial de Optuna para la optimización de los hiperparámetros. Omitimos el checkpoint progresivo del modelo y devolvemos directamente el mejor val_loss encontrado (tenemos un callback de EarlyStopping)
        Args:
            X_train (numpy.ndarray): Datos de entrada con forma (n_ejemplos, timesteps, n_features).
            y_train (numpy.ndarray): Etiquetas correspondientes con forma (n_ejemplos, 1).
            X_val (numpy.ndarray): Datos de entrada con forma (n_ejemplos, timesteps, n_features).
            y_val (numpy.ndarray): Etiquetas correspondientes con forma (n_ejemplos, 1).
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del lote durante el entrenamiento.
        Returns:
            best_val_loss (float): El mejor error de validación MSE en el momento de interrupción del entrenamiento
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience,  restore_best_weights=True, verbose=1)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0,
        )

        return history.history['val_loss'][early_stopping.stopped_epoch]

    def plot_training_history(self, history):
        """
        Visualiza gráficamente el historial de entrenamiento.

        Args:
            history (tf.keras.callbacks.History): Historial de entrenamiento devuelto por model.fit().
        """
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        metrics = ['loss', 'mean_absolute_error', 'root_mean_squared_error']
        val_metrics = ['val_loss', 'val_mean_absolute_error', 'val_root_mean_squared_error']
        names = ['MSE', 'MAE', 'RMSE']
        epochs = np.arange(1,len(history.history['loss'])+1,1)
        
        for i, (metric, val_metric, name) in enumerate(zip(metrics, val_metrics, names)):
            ax = axes[i]
            ax.plot(epochs,history.history[metric], label=f'Training {name}')
            ax.plot(epochs,history.history[val_metric], label=f'Validation {name}')
            ax.set_xticks(epochs)
            ax.set_title(f'Training and Validation {name}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(name)
            ax.legend()

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """
        Guarda el modelo entrenado en un archivo cuando se invoca el método en cualquier momento (diferente al callback checkpoint).

        Args:
            filepath (str): Ruta del archivo donde se guardará el modelo.
        """
        self.model.save(filepath)

    def load_model(self, filepath):
        """
        Carga un modelo previamente guardado desde un archivo.

        Args:
            filepath (str): Ruta del archivo desde donde se cargará el modelo.
        """
        self.model = tf.keras.models.load_model(filepath)  
    
    def predict(self, X):
        """
        Realiza predicciones utilizando el modelo entrenado.

        Args:
            X (numpy.ndarray): Datos de entrada con forma (n_ejemplos, timesteps, n_features).

        Returns:
            numpy.ndarray: Predicciones correspondientes con forma (n_ejemplos, 1).
        """
        return self.model.predict(X)