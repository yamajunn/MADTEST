import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# MNISTデータセットのロード
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))
x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))

# データサンプリング
sample_size = x_train.shape[0] // 8
x_train = x_train[:sample_size]
y_train = y_train[:sample_size]

# VAEのモデル定義
latent_dim = 2  # 潜在空間の次元数

def build_vae(latent_dim):
    # エンコーダー
    encoder_inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(encoder_inputs)
    x = layers.MaxPooling2D(padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling)([z_mean, z_log_var])

    # デコーダー
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x = layers.UpSampling2D()(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)

    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")

    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = models.Model(encoder_inputs, vae_outputs, name="vae")

    return vae, encoder, decoder

vae, encoder, decoder = build_vae(latent_dim)
vae.compile(optimizer='adam', loss='binary_crossentropy')

# VAEの訓練
vae.fit(x_train, x_train, epochs=10, batch_size=128, validation_split=0.1)

# 潜在空間のプロット
def plot_latent_space(encoder, x_test, y_test, figsize=(10, 10)):
    z_mean, _, _ = encoder.predict(x_test, batch_size=128)
    plt.figure(figsize=figsize)
    scatter = plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test, cmap='tab10', s=0.5)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.title("Latent Space")
    plt.colorbar(scatter, label='Digit Label')
    plt.show()

plot_latent_space(encoder, x_test[:sample_size], y_test[:sample_size])
