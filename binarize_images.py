import cv2
import os
import argparse
import numpy as np


def gerar_segmentacao(input_folder, output_folder):
    formatos_aceitos = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']

    if os.path.exists(input_folder):
        arquivos = os.listdir(input_folder)
    else:
        print('Diretorio de imagens nao foi encontrado')
        return False
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    
    for arquivo in arquivos:
        if arquivo.split('.')[-1] in formatos_aceitos: # Verifica se é imagem no formato aceito
            img = cv2.imread(input_folder+'/'+arquivo)
            imagem_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # Converte para HSV
            fator_saturacao = 10
            imagem_hsv[:, :, 1] = cv2.multiply(imagem_hsv[:, :, 1], fator_saturacao) # Aumenta a saturação das cores
            imagem_saturada = cv2.cvtColor(imagem_hsv, cv2.COLOR_HSV2BGR) # Retorna pro BGR
            green_filter = imagem_saturada[ : , : , 1] # Coleta o canal verde
            imagem_contraste = cv2.convertScaleAbs(green_filter, alpha=1.5, beta=-50) # Aumenta o contraste da imagem pra realçar ainda mais o verde
            _, tresh = cv2.threshold(imagem_contraste, 200, 255, cv2.THRESH_BINARY) # Thresholding para binearizar
            kernel = np.ones((3, 3), np.uint8)
            morph = cv2.morphologyEx(tresh, cv2.MORPH_CLOSE, kernel, iterations=3) # Aplicação de transformação morfologica para suavizar o resultado

            cv2.imwrite(output_folder+'/'+arquivo, morph)
    
    print('\nGeracao concluida\nOs arquivos finais possuem o mesmo nome dos originais\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processa imagens extraindo vegetacao")

    parser.add_argument('--input', required=True, help='Caminho para o diretorio com imagens')
    parser.add_argument('--output', required=True, help='Caminho para o diretorio de saida')

    args = parser.parse_args()
    
    gerar_segmentacao(args.input, args.output)