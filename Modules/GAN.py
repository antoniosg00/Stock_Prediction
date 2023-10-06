import tensorflow as tf
import pandas as pd
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Input, LeakyReLU, BatchNormalization, Dropout, MaxPooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import os
import time
import math

class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, decay_steps, decay_rate) -> None:
        self.initial_lr = initial_lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
    
    def __call__(self, step):
        return self.initial_lr * self.decay_rate ** (step / self.decay_steps)

class StockPriceGAN:
    """
    Este código implementa una Generative Adversarial Network (GAN) diseñada para predecir los precios futuros de acciones a partir de datos históricos. La GAN consta de un generador y un discriminador, que compiten entre sí para mejorar la calidad de las predicciones.

    El generador es un modelo LSTM que se entrena para generar secuencias de precios de acciones futuras basándose en datos de entrada históricos. El discriminador, por otro lado, es una red neuronal convolucional (CNN) que se entrena para distinguir entre datos reales y generados por el generador.

    El objetivo principal de este código es entrenar la GAN para que el generador aprenda a generar secuencias de precios de acciones que sean indistinguibles de los datos reales, mientras que el discriminador trata de discernir si los datos son reales o falsos. Este proceso de competencia entre el generador y el discriminador conduce a la mejora de las predicciones del generador.

    A medida que avanzan las épocas de entrenamiento, se espera que el generador sea capaz de producir predicciones precisas de los precios de las acciones para el día siguiente, lo que puede ser valioso en aplicaciones financieras y de inversión.
    
    Se siguen consejos de https://github.com/soumith/ganhacks
    """
    def __init__(self, n_features, timesteps, generator_layer_dims=[128,64], discriminator_conv_dims=[(32,2,1),(64,2,1),(128,2,1)], pool_size=2, discriminator_dense_dims=[200,200], momentum=0.9, alpha_lrelu=0.2, dropout_rate=0.5, initial_lr=0.01, decay_steps=100, decay_rate=1, kernel_initializer='glorot_uniform'):
        """
        Inicializa la clase StockPriceGAN con parámetros y modelos.
        
        Args:
            timesteps (int): Número de pasos de tiempo en los datos de entrada.
            n_features (int): Número de características en cada paso de tiempo.
            generator_lr (float): Tasa de aprendizaje para el generador.
            discriminator_lr (float): Tasa de aprendizaje para el discriminador.
        """
        # Establecemos las dimensiones adecuadas para los datos
        self.timesteps = timesteps
        self.n_features = n_features

        # Fijamos los hiperparámetros de las redes neuronales
        self.generator_layer_dims = generator_layer_dims 
        self.discriminator_conv_dims = discriminator_conv_dims
        self.discriminator_dense_dims = discriminator_dense_dims
        self.momentum = momentum # Hiperparámetro de las BatchNormalization
        self.alpha_lrelu = alpha_lrelu # Hiperparámetro de las capas Leaky-ReLU AÑADIR EN OPTUNA POSIBILIDAD DE SER 0 (RELU)
        self.dropout_rate = dropout_rate 
        self.pool_size = pool_size
        self.kernel_initializer=kernel_initializer # O 'glorot_uniform' o 'he_normal'

        # Fijamos los hiperparámetros de la tasa de aprendizaje
        lr_schedule = ExponentialDecay(initial_lr=initial_lr, decay_steps=decay_steps, decay_rate=decay_rate) 
        self.generator_optimizer = Adam(learning_rate=lr_schedule) # MISMO OPTIMIZADOR PARA G Y D. CUIDADO.
        self.discriminator_optimizer = Adam(learning_rate=lr_schedule)

        # Creamos los modelos
        self.generator = self.build_generator() # Generador (modelo LSTM)
        self.discriminator = self.build_discriminator() # Discriminador (modelo CNN)
        self.gan = self.build_gan() # La GAN completa (para entrenar al generador)

        # Define el directorio para guardar los checkpoints
        self.current_epoch = 0  # Para llevar un registro de las épocas
        self.checkpoint_dir = 'C:\\Users\\34722\\Desktop\\Stock_Prediction\\GAN_training_checkpoints'

        # Crea el directorio si no existe
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        # Define el prefijo del checkpoint con el número de época
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, f"ckpt")
        self.checkpoint_interval = 1  # Cada cuántas épocas guardamos el entrenamiento

        self.checkpoint = tf.train.Checkpoint(
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )

    def build_generator(self):
        """
        Construye el modelo del generador (LSTM).
        """
        generator = Sequential()
        generator.add(Input(shape=(self.timesteps, self.n_features)))  # Recordemos que keras ignora la primera dimensión (n_ejemplos) --> https://stackoverflow.com/questions/44747343/keras-input-explanation-input-shape-units-batch-size-dim-etc
        for i, units in enumerate(self.generator_layer_dims): # Una o dos capas (no más para evitar overfitting)
            if i==len(self.generator_layer_dims)-1: generator.add(LSTM(units, return_sequences=False, activation='tanh', kernel_initializer=self.kernel_initializer)) # En la última capa return_sequences=False para obtener 1 solo valor
            else: generator.add(LSTM(units, return_sequences=True, activation='tanh', kernel_initializer=self.kernel_initializer))
            generator.add(Dropout(self.dropout_rate))
        generator.add(Dense(1, kernel_initializer=self.kernel_initializer)) # Salida lineal para predecir el precio del día siguiente

        return generator

    def build_discriminator(self): # ¿AÑADIR PADDING SAME O VALID COMO HIPERPARÁMETRO?
        """
        Construye el modelo del discriminador (CNN).
        """
        discriminator = Sequential()
        discriminator.add(Input(shape=(self.timesteps+1, 1))) # El discriminador va a discernir la veracidad de una serie de precios 'Close' --> n_features=1
        for units, kernel_size, strides in self.discriminator_conv_dims:
            discriminator.add(Conv1D(units, kernel_size=kernel_size, strides=strides, activation=LeakyReLU(self.alpha_lrelu), padding='valid', kernel_initializer=self.kernel_initializer))
            discriminator.add(BatchNormalization(momentum=self.momentum))
            discriminator.add(MaxPooling1D(pool_size=self.pool_size, padding='same'))

        for units in self.discriminator_dense_dims:
            discriminator.add(Dense(units, activation=LeakyReLU(self.alpha_lrelu), kernel_initializer=self.kernel_initializer))
            discriminator.add(BatchNormalization(momentum=self.momentum))
            discriminator.add(Dropout(self.dropout_rate))
        discriminator.add(Dense(1, activation='sigmoid', kernel_initializer=self.kernel_initializer)) # Salida sigmoide para clasificación binaria (T o F)

        # discriminator.compile(loss='binary_crossentropy', optimizer=self.discriminator_optimizer) # Compilamos el discriminador

        return discriminator
    
    def build_gan(self):
        self.discriminator.trainable = False # Para que el el discriminador no se entrene mientras entrenamos el generador
        gan_input = Input(shape=(self.timesteps, self.n_features))
        generated_prices = self.generator(gan_input)
        gan_output = self.discriminator(tf.concat([gan_input[:, :, 0], generated_prices], axis=1))
        gan = Model(gan_input, gan_output)
        gan.compile(loss='binary_crossentropy', optimizer=self.generator_optimizer)
        return gan
    
    def train_discriminator(self, X_batch, y_batch):
        """
        Entrena el discriminador en datos reales y falsos utilizando GradientTape.

        Args:
            X_train (numpy.ndarray): Datos de entrada con forma (lote_size, timesteps, n_features).
            y_train (numpy.ndarray): Etiquetas correspondientes con forma (lote_size, 1).

        Returns:
            float: Pérdida combinada del discriminador.
        """
        batch_size = X_batch.shape[0]
        labels_real = tf.ones((batch_size, 1))
        labels_fake = tf.zeros((batch_size, 1))

        with tf.GradientTape() as tape:
            # Generar datos reales concatenados que incluyan el día siguiente
            real_data = tf.concat([X_batch[:, :, 0], y_batch], axis=1)
            # Calcular la salida del discriminador en datos reales
            d_output_real = tf.reshape(self.discriminator(real_data), shape=(-1,1))
            # Calcular la pérdida en datos reales
            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels_real, d_output_real))

            # Generar datos falsos con el generador
            generated_prices = self.generator(X_batch)
            fake_data = tf.concat([X_batch[:, :, 0], generated_prices], axis=1)
            # Calcular la salida del discriminador en datos falsos
            d_output_fake = tf.reshape(self.discriminator(fake_data), shape=(-1,1))
            # Calcular la pérdida en datos falsos
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(labels_fake, d_output_fake))

            # Pérdida total del discriminador
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Calcular los gradientes y aplicarlos al discriminador
        d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(d_gradients, self.discriminator.trainable_variables))

        return d_loss
    
    def train_generator_through_gan(self, X_batch):
        """
        Entrena el generador en los datos de entrada y etiquetas correspondientes utilizando GradientTape.

        Args:
            X_batch (numpy.ndarray): Datos de entrada con forma (lote_size, timesteps, n_features).
            y_batch (numpy.ndarray): Etiquetas correspondientes con forma (lote_size, 1).

        Returns:
            float: Pérdida del generador.
        """
        with tf.GradientTape() as tape:
            generated_prices = self.generator(X_batch)
            gan_output = self.discriminator(tf.concat([X_batch[:, :, 0], generated_prices], axis=1))
            gan_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(gan_output), gan_output))

        g_gradients = tape.gradient(gan_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

        return gan_loss
    
    # INTENTO FALLIDO DE ENTRENAMIENTO DE AMBOS MODELOS INDEPENDIENTEMENTE (NO TIENE SENTIDO)
    # def train_generator(self, X_batch, y_batch):
    #     """
    #     Entrena el generador en los datos de entrada y etiquetas correspondientes utilizando GradientTape.

    #     Args:
    #         X_batch (numpy.ndarray): Datos de entrada con forma (lote_size, timesteps, n_features).
    #         y_batch (numpy.ndarray): Etiquetas correspondientes con forma (lote_size, 1).

    #     Returns:
    #         float: Pérdida del generador.
    #     """
    #     with tf.GradientTape() as tape:
    #         # Generar precios falsos con el generador
    #         generated_prices = self.generator(X_batch)
    #         # Calcular la pérdida del generador utilizando el error cuadrático medio (MSE)
    #         g_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_batch, generated_prices))

    #     # Calcular los gradientes y aplicarlos al generador
    #     g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
    #     self.generator_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

    #     return g_loss
    
    def train(self, X_train, y_train, epochs, batch_size):
        """
        Entrena la GAN en los datos de entrada.

        Args:
            X_train (numpy.ndarray): Datos de entrada con forma (n_ejemplos, timesteps, n_features).
            y_train (numpy.ndarray): Etiquetas correspondientes con forma (n_ejemplos, 1).
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del lote durante el entrenamiento.
        """
        resto = X_train.shape[0] % batch_size 
        n_batches = math.ceil(X_train.shape[0] / batch_size)

        dictionary = {'Epoch': [], 'g_loss': [], 'd_loss': []}  # Almaceno errores por epoch para representarlos
        generated_df = pd.DataFrame()

        for epoch in range(epochs):

            start = time.time()
            self.current_epoch += 1
            # Listas para el cálculo de la pérdida media de la época:
            d_losses = [] 
            g_losses = []
            
            for batch in range(n_batches): # Recorro los lotes
                if batch == 0: # El primer batch es más pequeño, los demás ya todos iguales
                    X_batch = X_train[:resto]
                    y_batch = y_train[:resto]
                else:
                    X_batch = X_train[resto+(batch-1)*batch_size:resto+(batch-1)*batch_size + batch_size]
                    y_batch = y_train[resto+(batch-1)*batch_size:resto+(batch-1)*batch_size + batch_size]

                # Entrenar el discriminador
                d_loss = self.train_discriminator(X_batch, y_batch)
                d_losses.append(d_loss.numpy())

                # Entrenar el generador
                gan_input = X_batch
                gan_loss = self.train_generator_through_gan(gan_input)
                g_losses.append(gan_loss.numpy())

            # Imprimir el progreso del entrenamiento
            g_loss =sum(g_losses)/len(g_losses) # Sé que no todos los lotes son iguales (el primero no) pero es muy muy aproximado
            d_loss = sum(d_losses)/len(d_losses)
            print (f'Time for epoch {epoch + 1}/{epochs} is {(time.time()-start):.4f} s. Losses: G Loss = {g_loss:.8f} y D Loss = {d_loss:.5f}')
            dictionary['Epoch'].append(epoch+1)
            dictionary['g_loss'].append(g_loss)
            dictionary['d_loss'].append(d_loss)

            # Guardar el checkpoint al final de cada 'self.checkpoint_interval' épocas
            if (self.current_epoch) % self.checkpoint_interval == 0: # Punto de control del entrenamiento y modelos
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

                with open(self.checkpoint_dir+'\\reg_epoch.txt', 'w') as epochtxt: # Registramos la epoch guardada
                    epochtxt.write('Último epoch realizado: {}. \n'.format(self.current_epoch))

                generated_df[str(self.current_epoch)] = tf.reshape(self.generator(X_train), shape=(-1)) # Guardamos resultados generados

        return dictionary, generated_df