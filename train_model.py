import keras
import numpy as np
import cv2
import os
import argparse


def prepare_data(x_path, y_path):
    '''Faz a preparação dos dados dividindo em X, Y de treino e teste. Utiliza 20% como teste, escolhidos de forma aleatória.'''
    
    imagem_formato = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']
    
    x = []
    y = []
    
    if os.path.exists(x_path) and os.path.exists(y_path):
        
        imgs_x = os.listdir(x_path)
        imgs_y = os.listdir(y_path)

        for imagem in imgs_x:
            # verifica se é imagem no formato aceito e se possui um label equivalente
            # o processamento da imagem e do label juntos é para garantir a ordem para o treino
            if imagem.split('.')[-1] in imagem_formato and imagem in imgs_y:
                
                img_x = cv2.imread(x_path+'/'+imagem)
                img_x = prepare_x_input(img_x)
                img_x = img_x/255
                
                x.append(img_x)

                img_y = cv2.imread(y_path+'/'+imagem, 0)
                img_y = img_y/255
                
                y.append(img_y)
    else:
        print('Diretorio nao encontrado')
        return False
    
    x = np.array(x)
    y = np.array(y)

    qtd_teste = int(len(x) * 0.2) # Define a quantidade de amostras para o teste

    indices = np.random.permutation(len(x)) # Arranja os indices de forma aleatória. X e Y possuem indices espelhados para preservar a ordem das amostras e labels

    indices_teste = indices[:qtd_teste]
    x_test = x[indices_teste]
    y_test = y[indices_teste]

    indices_treino = indices[qtd_teste:]
    x_train = x[indices_treino]
    y_train = y[indices_treino]

    return x_train, y_train, x_test, y_test

def prepare_x_input(image):
    # Realçar características básicas da vegetação para auxiliar na segmentação realizada pela rede
    # Aumento de saturação para realçar as cores, seguido da seleção de canal verde

    fator_saturacao = 10

    imagem_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    imagem_hsv[:, :, 1] = cv2.multiply(imagem_hsv[:, :, 1], fator_saturacao)
    imagem_saturada = cv2.cvtColor(imagem_hsv, cv2.COLOR_HSV2BGR)
    green_filter = imagem_saturada[ : , : , 1]
    
    return green_filter


def create_modelo(input_shape):
    inputs = keras.layers.Input(input_shape)

    # Encoder - reduz as dimensões da imagem capturando as características
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck - compactação máxima antes da expansão (em tese todos padrões estão presentes aqui)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder - reconstrução da imagem e expansão da dimensão
    up4 = keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3)
    merge4 = keras.layers.concatenate([up4, conv2])
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge4)
    conv4 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up5 = keras.layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4)
    merge5 = keras.layers.concatenate([up5, conv1])
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(merge5)
    conv5 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    outputs =keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(conv5)

    modelo = keras.Model(inputs=[inputs], outputs=[outputs])

    return modelo
    


def treinar_rede(x_path, y_path, model_path):

    # Prepara os dados e faz a divisão em teste e treino
    print('\nPreparando dados...')
    x_train, y_train, x_test, y_test = prepare_data(x_path, y_path)

    # Cria o modelo da rede - Necessário a presença de GPU devido as camadas de convolução
    print('\nCriando modelo...')
    modelo = create_modelo((500,500,1))

    # Compila o modelo usando o erro Binary CrossEntropy - indicado para erro categórico (no caso 0 ou 1)
    modelo.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    # Faz o treinamento do modelo sobre os dados de treino e vai validando com os dados de teste para acompanhar o progresso
    # Atualiza os pesos a cada 10 amostras (melhor generalização com batchs maiores, mas o banco de dados é pequeno)
    # Treinamento de 100 épocas (relativamente pouco, mas como já ajudamos a rede realçando características de interesse, é o suficiente)
    # OBS: meu PC não possui GPU, logo realizei a implementação via instância T4 do Google Colab que possui uso limitado, por isso as 100 epocas também.
    print('\nIniciando treinamento\n')
    modelo.fit(x_train, y_train, batch_size=10, epochs=100, validation_data=(x_test, y_test))
    evaluate = modelo.evaluate(x_test, y_test)
    # Salva o modelo após o treinamento
    print('\n\nSalvando modelo')
    modelo.save(model_path)
    print('\nConcluido')
    print(f'\nErro: {evaluate[0]:.4f}\nAccuracy: {evaluate[1]:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Treina um modelo de CNN para segmentacao de vegetacao")

    parser.add_argument('--rgb', required=True, help='Caminho para o diretorio com imagens - input da rede')
    parser.add_argument('--groundtruth', required=True, help='Caminho para o diretorio de imagens segmentadas - labels da rede')
    parser.add_argument('--modelpath', required=True, help='Caminho de saida onde o modelo da rede sera salvo')

    args = parser.parse_args()

    treinar_rede(args.rgb, args.groundtruth, args.modelpath)
    
    



