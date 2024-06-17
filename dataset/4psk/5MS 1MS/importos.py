"""
Este script é responsável por importar dados IQ de arquivos .npy e plotar a constelação de cada arquivo
automaticamente no diretório onde o script está localizado. O script detecta automaticamente o diretório,
carrega os arquivos .npy, converte dados IQ de tuplas para complexos se necessário, e salva as imagens das 
constelações na mesma pasta dos arquivos .npy.

Como usar:
1. Coloque este script no diretório onde estão os arquivos .npy.
2. Execute o script.
3. As imagens das constelações serão salvas na mesma pasta com o mesmo nome dos arquivos .npy e extensão .png.

Exemplo:
Se o diretório contém arquivos:
- sent_data_complex.npy
- received_data_tuple.npy

Este script gerará imagens:
- sent_data_complex.png
- received_data_tuple.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Diretório onde o script está localizado
directory = os.path.dirname(os.path.abspath(__file__))
print(f"Verificando arquivos no diretório: {directory}")

# Função para carregar os dados de um arquivo .npy
def load_data(file_path):
    try:
        print(f"Carregando dados de {file_path}")
        data = np.load(file_path)
        print(f"Dados carregados, shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Erro ao carregar {file_path}: {e}")
        return None

# Função para converter dados IQ (tuplas) para complexos
def convert_to_complex(data):
    if data is not None and data.ndim == 2:  # Verifica se os dados estão em tuplas
        print("Convertendo dados IQ (tuplas) para complexos")
        data = data[:, 0] + 1j * data[:, 1]
    return data

# Função para plotar e salvar a constelação
def plot_and_save_constellation(data, title, save_path):
    if data is not None:
        plt.figure(figsize=(8, 8))
        plt.scatter(data.real, data.imag, color='blue', s=1)
        plt.title(title)
        plt.xlabel('Parte Real')
        plt.ylabel('Parte Imaginária')
        plt.grid(True)
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.savefig(save_path)
        plt.close()
        print(f"Imagem salva em {save_path}")
    else:
        print(f"Dados inválidos para {title}, constelação não gerada.")

# Verificar se o diretório existe
if not os.path.exists(directory):
    print(f"Erro: O diretório {directory} não existe.")
else:
    # Obter lista de arquivos .npy no diretório
    file_paths = [f for f in os.listdir(directory) if f.endswith('.npy')]
    print(f"Arquivos encontrados: {file_paths}")

    # Loop para carregar e plotar os dados de cada arquivo
    for file_path in file_paths:
        full_path = os.path.join(directory, file_path)
        data = load_data(full_path)
        data = convert_to_complex(data)  # Converter se os dados estiverem em tuplas
        save_path = os.path.join(directory, f'{os.path.splitext(file_path)[0]}.png')
        plot_and_save_constellation(data, f'Constelação {file_path}', save_path)
        print(f"Plotagem concluída para {file_path}")

