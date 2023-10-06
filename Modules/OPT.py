import optuna
import VAE
import LSTM
import numpy as np
import pandas as pd
import joblib
import math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import MeanSquaredError as MSE
from tensorflow.keras.callbacks import EarlyStopping


class OPT_VAE:
    """
    LOS PATH DEBEN TERMINAR EN \\ (WIN) Ó / (LINUX)
    """
    def __init__(self, X_train, X_val):
        self.X_train = X_train
        self.X_val = X_val
        self.study_df = pd.DataFrame(columns=['trial_number', 'loss', 'latent_dim', 'vae_layer_0_units', 'vae_layer_1_units', 'vae_layer_2_units', 'vae_layer_3_units', 'dropout_rate','alpha_lrelu', 'momentum', 'initial_lr', 'decay_steps', 'decay_rate', 'l1', 'l2', 'batch_size'])
        self.study = None # Creo el atributo para poder cargar modelos guardados

    def objective(self, trial):
        # Dimensiones
        n_examples = self.X_train.shape[0]
        original_dim=self.X_train.shape[1]

        # Tamaño de los lotes
        batch_size = trial.suggest_categorical("batch_size", [32, 64])
        steps_by_epoch = math.ceil(n_examples/batch_size)

        # Hiperparámetros a optimizar
        latent_dim = trial.suggest_int('latent_dim', 10, 50, step=10)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.6, step=0.1)
        kl_beta = 1 # Lo fijamos en 1. Si a cada trial es variable, falseamos la estadística
        alpha_lrelu = trial.suggest_float('alpha_lrelu', 0.0, 0.5, step=0.1) # Parámetro alfa de Leaky-ReLU
        momentum = trial.suggest_float('momentum', 0.8, 0.99, log=True)
        l1 = trial.suggest_float('l1', 2e-2, 5e-3)
        l2 = trial.suggest_float('l2', 2e-2, 5e-3)

        # Optimización de capas y dimensiones
        model_layer_dims=[]
        max_possible_layers = 4 # Modificar esta variable para probar con otros numeros de capas pero modificar el numero de columnas del csv
        number_layers = trial.suggest_int('number_layers', 2, max_possible_layers)
        for i in range(number_layers):
            model_layer_dims.append(trial.suggest_int(f"vae_layer_{i}_units", 10, 70, step=10))
        for i in range(number_layers, max_possible_layers):
            model_layer_dims.append(0)

        # Learning rate y sus hiperparámetros
        initial_lr = trial.suggest_float('initial_lr', 1e-4, 3e-3, log=True)
        decay_steps = trial.suggest_categorical('decay_steps', [steps_by_epoch//2, steps_by_epoch//3, steps_by_epoch])
        decay_rate = trial.suggest_float('decay_rate', 0.9, 0.9999, log=True)

        # Creamos el modelo VAE con los hiperparámetros sugeridos
        vae = VAE.VariationalAutoEncoder(original_dim=original_dim, latent_dim=latent_dim, layer_dims=model_layer_dims[:number_layers], dropout_rate=dropout_rate, momentum=momentum, kl_beta=kl_beta, alpha_lrelu=alpha_lrelu, l1=l1, l2=l2)

        # Compilamos el modelo
        lr_schedule = ExponentialDecay(initial_learning_rate=initial_lr,decay_steps=decay_steps, decay_rate=decay_rate)
        optimizer = Adam(learning_rate=lr_schedule)
        vae.compile(optimizer, loss=MSE())
        
        # Entrenamos el modelo y guardamos la métrica que deseas optimizar
        callback = EarlyStopping(monitor='val_loss', verbose=0, patience=10, restore_best_weights=True)
        history = vae.fit(self.X_train, self.X_train, epochs=200, batch_size=batch_size, callbacks=[callback], validation_data=(self.X_val, self.X_val), verbose=0)
        
        # Registramos los resultados del trial en el DataFrame 
        self.study_df.loc[trial.number] = [trial.number, history.history['val_loss'][callback.best_epoch], latent_dim] + model_layer_dims + [dropout_rate, alpha_lrelu, momentum, initial_lr, decay_steps, decay_rate, l1, l2, batch_size]

        # Devuelvemos el valor que deseamos optimizar
        return history.history['val_loss'][callback.best_epoch]
    
    def run_optimization(self, trials, study_number, path): # Introducimos número de studio para el posterior guardado
        self.study_number = study_number # Guardamos el numero como atributo por si se necesita continuar posteriormente con la optimización
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(self.objective, n_trials=trials)
        joblib.dump(self.study, f'{path}VAEstudy{study_number}.pkl') # Guardamos el estudio
        self.study_df.to_csv(f'{path}optimization_results{study_number}.csv', index=False) # Guardamos datos en csv
        self.load_attributes() # Cargamos los atributos tras la optimización

        return None
    
    def load_study(self, study_number, path): # Método para cargar estudios almacenados
        self.study_number = study_number # Guardamos el numero como atributo por si se necesita continuar posteriormente con la optimización
        self.study = joblib.load(f'{path}VAEstudy{study_number}.pkl')
        self.load_attributes() # Cargamos los atributos

        return None
    
    def resume_optimization(self, trials, path): # path de guardado
        if self.study == None: assert Exception('No hay ningún estudio cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        new = True if len(self.study_df)==0 else False # Nueva instancia donde el df está vacío

        # Continuamos la optimización con el estudio cargado
        self.study.optimize(self.objective, n_trials=trials)
        # Guardamos el estudio actualizado
        joblib.dump(self.study, f'{path}VAEstudy{self.study.number}.pkl')
        # Actualizamos atributos
        self.load_attributes()
        # Actualizamos el archivo CSV de entrenamiento
        if new:
            old_csv = pd.read_csv(f'{path}optimization_results{self.study_number}.csv') # Extraemos los datos que teníamos de optimizaciones anteriores
            self.study_df = pd.concat([self.study_df, old_csv]) # Los juntamos con los datos nuevos
            self.study_df.to_csv(f'{path}optimization_results{self.study_number}.csv', index=False) # Guardamos datos en csv
        else: self.study_df.to_csv(f'{path}optimization_results{self.study_number}.csv', index=False) # Los datos nuevos se han añadido al df que teníamos y sobreescribimos el CSV

        return None
    
    def load_attributes(self):
        if self.study == None: assert Exception('No hay ningún estudio cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        
        # Fijamos como atributos los mejores parámetros obtenidos de la optimización global
        self.trial = self.study.best_trial.number # El intento con mejor resultado
        self.value = self.study.best_value # El menor val_loss encontrado
        self.latent_dim = self.study.best_params['latent_dim']
        self.number_layers = self.study.best_params['number_layers']
        self.model_layer_dims = []
        for i in range(self.number_layers):
            self.model_layer_dims.append(self.study.best_params[f"vae_layer_{i}_units"])
        self.dropout_rate = self.study.best_params["dropout_rate"]
        self.alpha_lrelu = self.study.best_params["alpha_lrelu"]
        self.momentum = self.study.best_params["momentum"]
        self.initial_lr = self.study.best_params["initial_lr"]
        self.decay_steps = self.study.best_params["decay_steps"]
        self.decay_rate = self.study.best_params["decay_rate"]
        self.l1 = self.study.best_params["l1"]
        self.l2 = self.study.best_params["l2"]
        self.batch_size = self.study.best_params["batch_size"]

        return None

    def best_params(self, study_number=None, path=None):
        if self.study == None: assert Exception('No hay ningún estudio cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        
        print(f"\nBest hyperparameters (found in trial {self.trial}):\n")

        print("latent_dim: ", self.latent_dim)
        print("number_layers: ", self.number_layers)
        for i, units in enumerate(self.model_layer_dims):
            print(f"vae_layer_{i}_units: ", units)
        print("dropout_rate: ", self.dropout_rate)
        print("alpha_lrelu: ", self.alpha_lrelu)
        print("momentum: ", self.momentum)
        print("initial_lr: ", self.initial_lr)
        print("decay_steps: ", self.decay_steps)
        print("decay_rate: ", self.decay_rate)
        print("l1: ", self.l1)
        print("l2: ", self.l2)
        print("batch_size: ", self.batch_size)

        print("\nValor de la función objetivo:\n", self.value)

        if path!=None and study_number!=None:
            if isinstance(path, str) and isinstance(study_number, int):
                file_name = path+f'VAEbest_hyperparameters{study_number}.txt'
                with open(file_name, "w") as f:
                    f.write(f"\nBest hyperparameters (found in trial {self.trial}):\n")
                        
                    f.write("latent_dim: {}\n".format(self.latent_dim))
                    f.write("number_layers: {}\n".format(self.number_layers))
                    for i, units in enumerate(self.model_layer_dims):
                        f.write("vae_layer_{}_units: {}\n".format(i, units))
                    f.write("dropout_rate: {}\n".format(self.dropout_rate))
                    f.write("alpha_lrelu: {}\n".format(self.alpha_lrelu))
                    f.write("momentum: {}\n".format(self.momentum))
                    f.write("initial_lr: {}\n".format(self.initial_lr))
                    f.write("decay_steps: {}\n".format(self.decay_steps))
                    f.write("decay_rate: {}\n".format(self.decay_rate))
                    f.write("l1: {}\n".format(self.l1))
                    f.write("l2: {}\n".format(self.l2))
                    f.write("batch_size: {}\n".format(self.batch_size))

                    f.write("\nValor de la función objetivo: {}\n".format(self.value))

                print("La información se ha guardado en {}.".format(file_name))
            else: print('"path" debe ser una ruta valida de tipo str y "study_number" un int')

        return None
    
    def plot_importance(self):
        if self.study == None: assert Exception('No hay ningún estudio cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        optuna.visualization.plot_param_importances(self.study)

        return None
    


class OPT_LSTM:
    """
    LOS PATH DEBEN TERMINAR EN \\ (WIN) Ó / (LINUX)
    SIEMPRE REALIZAR (O CONTINUAR) UN ENTRENAMIENTO LOCAL DESPÚES DE UNO GLOBAL
    """
    def __init__(self, data_tensor, z):
        # Tensores de datos obtenidos del VAE
        self.data_tensor = data_tensor
        self.z = z

        # Los dataframes donde vamos a almacenar los datos obtenidos en los estudios de optimización
        self.study_global_df = pd.DataFrame(columns=['trial_number', 'loss', 'timesteps', 'layers_type', 'lstm_layer_0_units', 'lstm_layer_1_units', 
                                              'lstm_layer_2_units', 'dropout_rate', 
                                              'initial_lr', 'decay_steps', 'decay_rate', 'batch_size']) # DataFrame de hiperparámetros globales
        self.study_local_df = pd.DataFrame(columns=['trial_number', 'loss', 'momentum', 'kernel_initializer', 
                                              'l1', 'l2', 'bidirectional']) # DataFrame de hiperparámetros locales
        
        # La instancia del estudio
        self.global_study = None # Creo el atributo para poder cargar modelos guardados
        self.local_study = None

    def objective_global(self, trial):
        # Primero Optuna sugiere un número de timesteps (ejemplos considerados para hacer una predicción) y construimos dataset con función VAE2GANdata
        timesteps = trial.suggest_int('timesteps', 5, 20, 5)
        X_train, Y_train, X_val, Y_val, _,_,_,_ = VAE.VAE2GANdata(self.data_tensor, self.z, use_z=True, val_size = 0.1, test_size = 0.1, timesteps=timesteps, optimizing=True) # Creamos el dataset completo para el GAN o LSTM con z y el dataset original y lo escalamos

        # Optuna sugiere valores para los hiperparámetros globales:
        layers_type = trial.suggest_categorical('layers_type', ['LSTM', 'GRU'])
        max_possible_layers = 3 # Modificar esta variable para probar con otros numeros de capas pero modificar el numero de columnas del csv
        number_layers = trial.suggest_int('number_layers', 2, max_possible_layers)
        model_layer_dims = []
        for i in range(number_layers):
            model_layer_dims.append(trial.suggest_int(f"lstm_layer_{i}_units", 25, 200, 25))
        for i in range(number_layers, max_possible_layers):
            model_layer_dims.append(0)
        dropout_rate = trial.suggest_float("dropout_rate", 0.3, 0.6, step=0.1)
        initial_lr = trial.suggest_float("initial_lr", 1e-4, 3e-3, log=True)
        decay_steps = trial.suggest_int("decay_steps", 100, 300, 25)
        decay_rate = trial.suggest_float("decay_rate", 0.9, 0.9999, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64])

        # Crear y entrenar modelo con los hiperparámetros globales sugeridos. Los locales dejo los prefijados en la propia clase StockPriceLSTM (son razonables)
        model = LSTM.StockPriceLSTM(
            n_features=X_train.shape[2],
            timesteps=X_train.shape[1],
            layers_type=layers_type,
            model_layer_dims=model_layer_dims[:number_layers],
            dropout_rate=dropout_rate,
            initial_lr=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            momentum=0.9, # A partir de estos los optimizo en la segunda fase (locales)
            kernel_initializer='glorot_uniform',
            l1_factor=0.0015,
            l2_factor=0.0015,
            bidirectional=False
        )

        val_loss = model.train_opt(X_train, Y_train, X_val, Y_val, epochs=10, batch_size=batch_size, patience=5)

        # Registramos los resultados del trial en el DataFrame de hiperparámetros globales
        self.study_global_df.loc[trial.number] = [trial.number, val_loss, timesteps, layers_type] + model_layer_dims + [dropout_rate, initial_lr, decay_steps, decay_rate, batch_size]

        return val_loss
    
    def objective_local(self, trial):
        X_train, Y_train, X_val, Y_val, _,_,_,_ = VAE.VAE2GANdata(self.data_tensor, self.z, use_z=True, val_size = 0.1, test_size = 0.1, timesteps=self.timesteps, optimizing=True) # Optimizing=True para silenciar prints
        momentum = trial.suggest_float("momentum", 0.8, 0.99, log=True)
        kernel_initializer = trial.suggest_categorical("kernel_initializer", ["glorot_uniform", "he_normal"])
        l1 = trial.suggest_float("l1", 1e-4, 1e-1)
        l2 = trial.suggest_float("l2", 1e-4, 1e-1)
        bidirectional = trial.suggest_categorical("bidirectional", [True, False])

        # Utilizar los hiperparámetros globales óptimos en la optimización local
        model = LSTM.StockPriceLSTM(
            n_features=X_train.shape[2],
            timesteps=X_train.shape[1],
            layers_type=self.layers_type,
            model_layer_dims=self.model_layer_dims,
            dropout_rate=self.dropout_rate,
            initial_lr=self.initial_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            momentum=momentum, # A partir de aquí son los que estoy optimizando
            kernel_initializer=kernel_initializer,
            l1_factor=l1,
            l2_factor=l2,
            bidirectional=bidirectional,
        )

        val_loss = model.train_opt(X_train, Y_train, X_val, Y_val, epochs=10, batch_size=self.batch_size, patience=5)

        # Registramos los resultados del trial en el DataFrame de hiperparámetros locales
        self.study_local_df.loc[trial.number] = [trial.number, val_loss, momentum, kernel_initializer, l1, l2, bidirectional]

        return val_loss
    
    def optimize_global_hyperparameters(self, n_trials, study_number, path):
        self.study_number = study_number # Guardamos el numero como atributo por si se necesita continuar posteriormente con la optimización
        self.global_study = optuna.create_study(direction='minimize')
        self.global_study.optimize(self.objective_global, n_trials=n_trials)
        joblib.dump(self.global_study, f'{path}LSTM_global_study{study_number}.pkl') # Guardamos el estudio
        self.study_global_df.to_csv(f'{path}global_optimization_results{study_number}.csv', index=False) # Guardamos datos en csv
        self.load_global_attributes()

        return None
    
    def optimize_local_hyperparameters(self, n_trials, path):
        self.local_study = optuna.create_study(direction='minimize')
        self.local_study.optimize(self.objective_local, n_trials=n_trials)
        joblib.dump(self.global_study, f'{path}LSTM_local_study{self.study_number}.pkl') # Guardamos el estudio
        self.study_local_df.to_csv(f'{path}local_optimization_results{self.study_number}.csv', index=False) # Guardamos datos en csv
        self.load_local_attributes()

        return None

    def run_optimization(self, global_trials, local_trials, study_number, path):
        self.optimize_global_hyperparameters(global_trials, study_number, path)
        self.optimize_local_hyperparameters(local_trials, path)
        
        return None

    def load_global_study(self, study_number, path): # Si solo se quiere cargar el estudio global
        self.study_number = study_number # Guardamos el numero como atributo por si se necesita continuar posteriormente con la optimización
        self.global_study = joblib.load(f'{path}LSTM_global_study{study_number}.pkl')
        self.load_global_attributes() # Cargamos los atributos

        return None
    
    def load_local_study(self, study_number, path): # Si solo se quiere cargar el estudio local
        self.study_number = study_number # Guardamos el numero como atributo por si se necesita continuar posteriormente con la optimización
        self.local_study = joblib.load(f'{path}LSTM_local_study{study_number}.pkl')
        self.load_local_attributes() # Cargamos los atributos

        return None

    def load_studies(self, study_number, path): # Método para cargar ambos estudios almacenados
        self.load_global_study(study_number, path)
        self.load_local_study(study_number, path)

        return None
    
    def resume_global_optimization(self, trials, path): # path de guardado
        if self.global_study == None: assert Exception('No hay ningún estudio global cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        new = True if len(self.study_global_df)==0 else False # Nueva instancia donde el df está vacío por si hemos reiniciado kernel

        # Continuamos la optimización con el estudio cargado
        self.global_study.optimize(self.objective_global, n_trials=trials)
        # Guardamos el estudio actualizado
        joblib.dump(self.global_study, f'{path}LSTM_global_study{self.study_number}.pkl')
        # Actualizamos atributos
        self.load_global_attributes()
        # Actualizamos el archivo CSV de entrenamiento
        if new:
            old_csv = pd.read_csv(f'{path}global_optimization_results{self.study_number}.csv') # Extraemos los datos que teníamos de optimizaciones anteriores
            self.study_global_df = pd.concat([self.study_global_df, old_csv]) # Los juntamos con los datos nuevos
            self.study_global_df.to_csv(f'{path}global_optimization_results{self.study_number}.csv', index=False) # Guardamos datos en csv
        else: self.study_global_df.to_csv(f'{path}global_optimization_results{self.study_number}.csv', index=False) # Los datos nuevos se han añadido al df que teníamos y sobreescribimos el CSV

        return None
    
    def resume_local_optimization(self, trials, path): # path de guardado
        if self.local_study == None: assert Exception('No hay ningún estudio local cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        new = True if len(self.study_local_df)==0 else False # Nueva instancia donde el df está vacío por si hemos reiniciado kernel

        # Continuamos la optimización con el estudio cargado
        self.local_study.optimize(self.objective_local, n_trials=trials)
        # Guardamos el estudio actualizado
        joblib.dump(self.local_study, f'{path}LSTM_global_study{self.study_number}.pkl')
        # Actualizamos atributos
        self.load_local_attributes()
        # Actualizamos el archivo CSV de entrenamiento
        if new:
            old_csv = pd.read_csv(f'{path}local_optimization_results{self.study_number}.csv') # Extraemos los datos que teníamos de optimizaciones anteriores
            self.study_local_df = pd.concat([self.study_local_df, old_csv]) # Los juntamos con los datos nuevos
            self.study_local_df.to_csv(f'{path}local_optimization_results{self.study_number}.csv', index=False) # Guardamos datos en csv
        else: self.study_local_df.to_csv(f'{path}local_optimization_results{self.study_number}.csv', index=False) # Los datos nuevos se han añadido al df que teníamos y sobreescribimos el CSV

        return None
    
    def load_global_attributes(self):
        if self.global_study == None: assert Exception('No hay ningún estudio global cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        # Fijamos como atributos los mejores parámetros obtenidos de la optimización global
        self.global_trial = self.global_study.best_trial.number # El intento con mejor resultado
        self.global_value = self.global_study.best_value # El menor val_loss encontrado
        self.timesteps = self.global_study.best_params['timesteps'] # El timesteps considerado
        
        self.layers_type = self.global_study.best_params['layers_type']
        self.number_layers = self.global_study.best_params['number_layers']
        self.model_layer_dims = []
        for i in range(self.number_layers):
            self.model_layer_dims.append(self.global_study.best_params[f"lstm_layer_{i}_units"])
        self.dropout_rate = self.global_study.best_params["dropout_rate"]
        self.initial_lr = self.global_study.best_params["initial_lr"]
        self.decay_steps = self.global_study.best_params["decay_steps"]
        self.decay_rate = self.global_study.best_params["decay_rate"]
        self.batch_size = self.global_study.best_params["batch_size"]

        return None
    
    def load_local_attributes(self):
        if self.local_study == None: assert Exception('No hay ningún estudio local cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        # Fijamos como atributos los mejores parámetros obtenidos de la optimización global
        self.local_trial = self.local_study.best_trial.number # El intento con mejor resultado
        self.local_value = self.local_study.best_value # El menor val_loss encontrado
        
        self.momentum = self.local_study.best_params['momentum']
        self.kernel_initializer = self.local_study.best_params['kernel_initializer']
        self.l1 = self.local_study.best_params['l1']
        self.l2 = self.local_study.best_params['l2']
        self.bidirectional = self.local_study.best_params['bidirectional']

        return None

    def best_params(self, study_number=None, path=None):
        if self.global_study == None: assert Exception('Por favor, carga los estudios global y local o ejecuta una optimización nueva.')
        if self.local_study == None: assert Exception('Por favor, carga los estudios global y local o ejecuta una optimización nueva.')
        print(f"\nBest hyperparameters (found in trials (global y local): {self.global_trial} y {self.local_trial}):\n")
        print("Global hyperparameters:\n")
        print("timesteps: ", self.timesteps)
        print("layers_type: ", self.layers_type)
        print("number_layers: ", self.number_layers)
        for i, units in enumerate(self.model_layer_dims):
            print(f"lstm_layer_{i}_units: ", units)
        print("dropout_rate: ", self.dropout_rate)
        print("initial_lr: ", self.initial_lr)
        print("decay_steps: ", self.decay_steps)
        print("decay_rate: ", self.decay_rate)
        print("batch_size: ", self.batch_size)

        print("\nLocal hyperparameters:\n")
        print("momentum: ", self.momentum)
        print("kernel_initializer: ", self.kernel_initializer)
        print("l1: ", self.l1)
        print("l2: ", self.l2)
        print("bidirectional: ", self.bidirectional)

        print("\nValor de la función objetivo:\n", self.local_value) # Esta pérdida es la global ya que he usado en la optimización los best_params de la global

        if path!=None and study_number!=None:
            if isinstance(path, str) and isinstance(study_number, int):
                file_name = path+f'LSTMbest_hyperparameters{study_number}.txt'
                with open(file_name, 'w') as f:
                    f.write("\nBest hyperparameters (found in trials (global y local): {} y {}):\n".format(self.global_trial, self.local_trial))
                    
                    f.write("timesteps: {}\n".format(self.timesteps))
                    f.write("layers_type: {}\n".format(self.layers_type))
                    f.write("number_layers: {}\n".format(self.number_layers))
                    for i, units in enumerate(self.model_layer_dims):
                        f.write("lstm_layer_{}_units: {}\n".format(i, units))
                    f.write("dropout_rate: {}\n".format(self.dropout_rate))
                    f.write("initial_lr: {}\n".format(self.initial_lr))
                    f.write("decay_steps: {}\n".format(self.decay_steps))
                    f.write("decay_rate: {}\n".format(self.decay_rate))
                    f.write("batch_size: {}\n".format(self.batch_size))
                    
                    f.write("momentum: {}\n".format(self.momentum))
                    f.write("kernel_initializer: {}\n".format(self.kernel_initializer))
                    f.write("l1: {}\n".format(self.l1))
                    f.write("l2: {}\n".format(self.l2))
                    f.write("bidirectional: {}\n".format(self.bidirectional))
                    
                    f.write("\nValor de la función objetivo:\n{}".format(self.local_value))

                print("La información se ha guardado en {}.".format(file_name))
            else: print('"path" debe ser una ruta valida de tipo str y "study_number" un int')

        return None

    def plot_global_importance(self): 
        if self.global_study == None: assert Exception('No hay ningún estudio global cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        optuna.visualization.plot_param_importances(self.global_study)

        return None
    
    def plot_local_importance(self):
        if self.local_study == None: assert Exception('No hay ningún estudio local cargado. Por favor, carga un estudio o ejecuta una optimización nueva.')
        optuna.visualization.plot_param_importances(self.local_study)

        return None
    
    def plot_importance(self):  # ARREGLAR ESTAS FUNCIONES PUES NO ME SACA DIBUJO. 
                                # SI USO optuna.visualization.plot... DESDE FUERA, SÍ FUNCIONA.
        self.plot_global_importance()
        self.plot_local_importance()

        return None