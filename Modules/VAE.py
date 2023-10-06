import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.regularizers import L1L2
from sklearn.preprocessing import MinMaxScaler

# Fijar las semillas aleatorias para tener reproducibilidad
np.random.seed(0) # Para numpy
tf.random.set_seed(0) # Para Tensorflow

@tf.keras.saving.register_keras_serializable()
class Sampling(tf.keras.layers.Layer):
    # Usamos z_mean y z_log_var para obtener z, el vector codificado en el espacio latente
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
@tf.keras.saving.register_keras_serializable()
class Encoder(tf.keras.layers.Layer):
    # Codificador para obtener (z_mean, z_log_var, z)

    def __init__(self, latent_dim=15, layer_dims=[128, 64], dropout_rate=0.5, momentum=0.99, name='encoder', alpha_lrelu=0.2, l1=0.1, l2=0.1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_layers = [Dense(dim, activation=LeakyReLU(alpha=alpha_lrelu), kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=l1, l2=l2)) for dim in layer_dims] # Capa lineal con inicialización Xavier
        self.norm_layers = [BatchNormalization(momentum=momentum) for _ in layer_dims] # Normalización del lote
        self.dropout_layers = [Dropout(dropout_rate) for _ in layer_dims] # Capas Dropout
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        x = inputs
        for dense_layer, norm_layer, dropout_layer in zip(self.dense_layers, self.norm_layers, self.dropout_layers):
            x = dense_layer(x)
            x = norm_layer(x)
            x = dropout_layer(x)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z

@tf.keras.saving.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    # Descodificamos z de vuelta al espacio original

    def __init__(self, original_dim, layer_dims=[128, 64], dropout_rate=0.5, momentum=0.99, name='decoder', alpha_lrelu=0.2, l1=0.1, l2=0.1, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dense_layers = [Dense(dim, activation=LeakyReLU(alpha=alpha_lrelu), kernel_initializer='glorot_uniform', kernel_regularizer=L1L2(l1=l1, l2=l2)) for dim in layer_dims[::-1]] # Invertimos el orden de las capas respecto al Encoder
        self.norm_layers = [BatchNormalization(momentum=momentum) for _ in layer_dims]
        self.dropout_layers = [Dropout(dropout_rate) for _ in layer_dims]
        self.dense_out = Dense(original_dim) 

    def call(self, inputs):
        x = inputs
        for dense_layer, norm_layer, dropout_layer in zip(self.dense_layers, self.norm_layers, self.dropout_layers):
            x = dense_layer(x)
            x = norm_layer(x)
            x = dropout_layer(x)
        return self.dense_out(x)

@tf.keras.saving.register_keras_serializable()
class VariationalAutoEncoder(tf.keras.Model):
    # Combinamos encoder y decoder

    def __init__(self, original_dim, latent_dim=15, layer_dims=[128, 64], dropout_rate=0.5, momentum=0.99, kl_beta=1, alpha_lrelu=0.2, l1=0.1, l2=0.1, name='VAE', **kwargs):
        super().__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, layer_dims=layer_dims, dropout_rate=dropout_rate, momentum=momentum, alpha_lrelu=alpha_lrelu, l1=l1, l2=l2)
        self.decoder = Decoder(original_dim=original_dim, layer_dims=layer_dims, dropout_rate=dropout_rate, momentum=momentum, alpha_lrelu=alpha_lrelu, l1=l1, l2=l2)
        self.kl_beta = kl_beta

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1) # KL_loss = -0.5 * sum(1 + log(variance) - mean^2 - variance)
        self.add_loss(self.kl_beta*kl_loss)
        return reconstructed
    
def VAE2GANdata(data_tensor, z, use_z=True, val_size = 0.1, test_size = 0.1, timesteps=10, optimizing=False):
    """
    Función del módulo VAE que genera, a partir de la salida del VAE, el dataset necesario para la siguiente fase del proyecto: la GAN o el modelo LSTM o GRU.
    Args:
        data_tensor (tf.Tensor): Datos de entrada con forma (n_ejemplos, n_features).
        z (tf.Tensor): Datos del espacio latente con forma (n_ejemplos, latent_dim).
        use_z (bool)
        val_size (float): Etiquetas correspondientes con forma (n_ejemplos, 1).
        test_size (float): Número de épocas de entrenamiento.
        timesteps (int): Ejemplos observados para hacer cada predicción
        optimizing (bool): if optimizing=True no se muestran los prints
    """
    # Elegimos si queremos integrar en el dataset los resultados del VAE obtenidos del espacio latente, i.e., el tensor z
    if use_z: concatenated_tensor = tf.concat([data_tensor, z], axis=1) # Creamos tensor unificado (n_examples, n_features)
    else: concatenated_tensor=data_tensor

    # Creamos tensores train, val, test (n_examples_set, n_features)
    train_tensor, val_tensor, test_tensor = np.array(concatenated_tensor[:round((1-val_size-test_size)*len(concatenated_tensor))]), np.array(concatenated_tensor[round((1-val_size-test_size)*len(concatenated_tensor)):round((1-test_size)*len(concatenated_tensor))]), np.array(concatenated_tensor[round((1-test_size)*len(concatenated_tensor)):])
    if not optimizing: print(f'Train tensor shape: {train_tensor.shape}, Validation tensor shape: {val_tensor.shape}, Test tensor shape: {test_tensor.shape}\n')

    def XYtensors(tensor, timesteps):
        n_examples = tensor.shape[0] - timesteps # p.e., si tengo 20 datos y observo los últimos 3 y predigo el siguiente, puedo hacer 15 predicciones etiquetadas.
        n_features = tensor.shape[1]

        X = np.empty((n_examples, timesteps, n_features)) # Solo una característica (el valor)
        Y = np.empty((n_examples, 1)) # Solo un día y solo una característica (el valor)

        for i in range(n_examples):
            for j in range(n_features):
                X[i,:,j] = tensor[i:i+timesteps,j]
            Y[i,0] = tensor[i+timesteps:i+timesteps+1,0] # Tomamos la primera feature (precio Close) del día siguiente
        return X, Y
    
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test) = XYtensors(train_tensor, timesteps), XYtensors(val_tensor, timesteps), XYtensors(test_tensor, timesteps)

    # Usamos consejos de https://stackoverflow.com/questions/71544858/valueerror-with-minmaxscaler-inverse-transform
    # Puede parecer enrevesado pero tiene buenos resultados
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    X_train, Y_train = scaler_X.fit_transform(X_train.reshape(-1,train_tensor.shape[1])).reshape(train_tensor.shape[0]-timesteps, timesteps, train_tensor.shape[1]), scaler_Y.fit_transform(Y_train)
    X_val, Y_val = scaler_X.transform(X_val.reshape(-1,val_tensor.shape[1])).reshape(val_tensor.shape[0]-timesteps, timesteps, val_tensor.shape[1]), scaler_Y.transform(Y_val)
    X_test, Y_test = scaler_X.transform(X_test.reshape(-1,test_tensor.shape[1])).reshape(test_tensor.shape[0]-timesteps, timesteps, test_tensor.shape[1]), scaler_Y.transform(Y_test)
    
    if not optimizing:
        print('Transformed tensor for next process:\n')
        print(f'X_train shape: {X_train.shape}, Y_train shape: {Y_train.shape}')
        print(f'X_val shape: {X_val.shape}, y_val shape: {Y_val.shape}')
        print(f'X_test shape: {X_test.shape}, Y_test shape: {Y_test.shape}')

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, scaler_X, scaler_Y