# Bibliotecas necessárias (execução local)

- pandas
- glob
- tensorflow
- sklearn

# diretórios e arquivos

|__ data: dados de normalidade e falha, utilizados para treinamento
|__ modelos: modelos CNN treinados, específicos de cada falha

CNN_TE.ipynb: Jupyter notebook que contém os códigos para definição/treinamento de uma CNN para detecção de falhas.

CNN_classificador_falhas: Jupyter notebook que contém os códigos para definição/treinamento de uma CNN para detecção de falhas.

Os códigos dos notebooks podem ser acessados e executados no google colab: [CNN_TE](https://colab.research.google.com/drive/1cZqxuuWkf4NO_zb1f8YrK1taM_w6FFzM?usp=sharing) e [CNN_classificador_falhas](https://colab.research.google.com/drive/1MOUrZVSVfkWDnn_6qI0OgwtvgozlgZdf?usp=sharing)

# Carregando um modelo salvo

``import tensorflow as tf``

``modelo = tf.keras.models.load_model('modelos/f1.h5')``