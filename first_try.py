import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Configurações iniciais
latent_dim = 50
epochs = 100
batch_size = 3200

# Funções auxiliares e modelos
def complex_to_real_imag(data_complex):
    return np.stack([np.real(data_complex), np.imag(data_complex)], axis=-1)

def build_encoder(latent_dim):
    inputs = layers.Input(shape=(2,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(16, activation='relu')(x)
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    model = models.Model(inputs, [z_mean, z_log_var], name='encoder')
    return model, z_mean, z_log_var
# A sequência de camadas densas com números decrescentes de unidades (256, 128, 64, 32) indica uma redução progressiva da dimensionalidade. 
# Cada camada usa a função de ativação ReLU ('relu'), que ajuda a introduzir não-linearidades no modelo e a combater o problema do gradiente desaparecendo.
# saídas do encoder: z_mean e z_log_var. Ambas são vetores de dimensão latent_dim, representando a média e o logaritmo da variância (log-variance) do espaço latente aprendido, respectivamente. 
# Estes são conceitos-chave em VAEs, permitindo a geração de novos dados através da amostragem estocástica.

def build_decoder(latent_dim):
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation='relu')(latent_inputs)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(2, activation='linear')(x)
    model = models.Model(latent_inputs, outputs, name='decoder')
    return model


def build_discriminator():
    inputs = layers.Input(shape=(2,)) #indica que cada amostra de entrada possui duas dimensões, que podem representar, por exemplo, a parte real e imaginária de um sinal complexo.
    x = layers.Dense(64, activation='leaky_relu')(inputs)
    x = layers.Dense(128, activation='leaky_relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs, name='discriminator')
    return model

def sampling(args):
    z_mean, z_log_var = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon
# Esta função não faz parte da arquitetura GAN, mas é essencial para os VAEs, permitindo que o modelo gere novas amostras a partir do espaço latente.
# Gera amostras aleatórias ε da distribuição normal padrão e as usa para amostrar do espaço latente, conforme definido na equação de reparametrização dos VAEs. 
# tf.exp(0.5 * z_log_var) calcula o desvio padrão. O resultado é uma amostra do espaço latente que pode ser usada pelo decoder para gerar novas amostras de dados.

def custom_iq_loss(y_true, y_pred):
    y_true_complex = tf.complex(y_true[:, 0], y_true[:, 1])
    y_pred_complex = tf.complex(y_pred[:, 0], y_pred[:, 1])
    error_magnitude = tf.abs(y_true_complex - y_pred_complex)   
    return tf.reduce_mean(tf.square(error_magnitude))

# A função custom_iq_loss é uma função de perda customizada, projetada especificamente para trabalhar com sinais IQ.
# Conversão para Complexo: Primeiro, a função converte os vetores de entrada y_true e y_pred, que representam os sinais verdadeiros e previstos respectivamente, 
#de suas representações de parte real e imaginária para números complexos. Isso é feito usando tf.complex, que combina as partes real e imaginária em um único número complexo.
# Calcula a magnitude do erro entre os sinais verdadeiros e previstos. Isso é efetivamente a distância euclidiana no espaço complexo, que leva em conta tanto a amplitude quanto a fase dos sinais.
# Cálculo da Perda: calcula a média do quadrado das magnitudes dos erros para todos os exemplos no batch. Isso proporciona uma única medida escalar da perda, que o TensorFlow pode minimizar durante o treinamento.
 

# Preparação dos dados
IQ_x_complex = np.load('sent_adjusted.npy')
IQ_y_complex = np.load('received_adjusted.npy')
IQ_x = complex_to_real_imag(IQ_x_complex)
IQ_y = complex_to_real_imag(IQ_y_complex)

# Normalização dos dados
IQ_x_normalized = (IQ_x - np.mean(IQ_x, axis=0)) / np.std(IQ_x, axis=0)
IQ_y_normalized = (IQ_y - np.mean(IQ_y, axis=0)) / np.std(IQ_y, axis=0)

# Criando um tf.data.Dataset
dataset = tf.data.Dataset.from_tensor_slices((IQ_x_normalized, IQ_y_normalized))
dataset = dataset.shuffle(10000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE).cache()

# Construindo o modelo
encoder, z_mean, z_log_var = build_encoder(latent_dim)
decoder = build_decoder(latent_dim)
discriminator = build_discriminator()

# Modelo VAE
inputs = layers.Input(shape=(2,))
z_mean, z_log_var = encoder(inputs)
z = layers.Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
reconstructed = decoder(z)
vae = models.Model(inputs, reconstructed, name='vae')

# Compilando o VAE
# Compilando o VAE com a função de perda customizada
vae.compile(optimizer='adam', loss=custom_iq_loss)

# Treinamento do VAE
print("Iniciando o treinamento do VAE...")
vae.fit(dataset, epochs=epochs, verbose=2)
print("Treinamento do VAE concluído.")


# Função para plotar o diagrama de constelação
def plot_constellation(ax, data, title):
    ax.scatter(data.real, data.imag, s=1, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('Parte Real')
    ax.set_ylabel('Parte Imaginária')
    ax.grid(True)

    # Gerar ruído para nova amostra de sinais
noise = np.random.normal(size=(IQ_x.shape[0], latent_dim))  # Reduzir o número de pontos para melhor visualização
fake_y = decoder.predict(noise)

real_x_complex = IQ_x[:, 0] + 1j * IQ_x[:, 1]
real_y_complex = IQ_y[:, 0] + 1j * IQ_y[:, 1]
fake_y_complex = fake_y[:, 0] + 1j * fake_y[:, 1]

fig, axs = plt.subplots(1, 3, figsize=(18, 6))
plot_constellation(axs[0], real_x_complex, "Constelação do Sinal Enviado")
plot_constellation(axs[1], real_y_complex, "Constelação do Sinal Recebido")
plot_constellation(axs[2], fake_y_complex, "Constelação do Sinal Aprendido pelo GAN")

plt.tight_layout()
plt.show()

def calculate_evm(transmitted_signal, received_signal):
    # Calcular o vetor de erro
    error_vector = received_signal - transmitted_signal

    # Calcular a magnitude do vetor de erro
    error_magnitude = np.abs(error_vector)

    # Calcular a potência média dos símbolos transmitidos
    average_power = np.mean(np.abs(transmitted_signal)**2)

    # Calcular EVM
    EVM = np.sqrt(np.mean(error_magnitude**2) / average_power)

    # Converter EVM para porcentagem
    EVM_percentage = EVM * 100

    # Convertendo EVM para dB
    EVM_dB = 20 * np.log10(EVM)

    return EVM_percentage, EVM_dB

# Suponha que 'real_x_complex' seja o sinal transmitido e 'fake_y_complex' seja o sinal gerado pelo GAN
EVM_percentage_new, EVM_dB_new = calculate_evm(real_x_complex, fake_y_complex)
EVM_percentage, EVM_dB = calculate_evm(IQ_x_complex, IQ_y_complex)

# Imprima o EVM
print("EVM (%):", EVM_percentage)
print("EVM (dB):", EVM_dB)

# Imprima o EVM
print("EVM_new (%):", EVM_percentage_new)
print("EVM_new (dB):", EVM_dB_new)

