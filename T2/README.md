# diretórios

|__ data: dados de normalidade e falha, utilizados para treinamento
|__ modelos: modelos CNN treinados, específicos de cada falha

# Carregando um modelo salvo

``import tensorflow as tf``

``modelo = tf.keras.models.load_model('modelos/f1.h5')``