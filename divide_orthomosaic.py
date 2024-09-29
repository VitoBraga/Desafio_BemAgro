import os
import cv2
import numpy as np
import argparse

def cortar_imagem(img_path, output_path):
    # Verifica se a imagem existe
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
    else:
        print('O caminho da imagem não existe')
        return False
    
    # Cria o diretório se precisar
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print('\nIniciando divisao...')
    altura_original = img.shape[0]
    comprimento_original = img.shape[1]

    # tamanhos arbitrários
    altura = 500
    comprimento = 500
    
    columns = int(comprimento_original / comprimento)
    rows = int(altura_original / altura)
    
    # Percorre a imagem com passos de 500px, nas bordas da imagem completa com preto (pensando no momento da segmentação, o valor 0 já estará atribuído)
    for i in range(rows+1):
        for j in range(columns+1):
            sub_img = img[i*altura : (i+1)*altura, j*comprimento : (j+1)*comprimento, :]
            if sub_img.shape[0] != 500 or sub_img.shape[1] != 500:
                new_area = np.zeros((altura, comprimento, 3), dtype=np.uint8)
                new_area[:sub_img.shape[0], : sub_img.shape[1] ] = sub_img
                new_area[ sub_img.shape[0]: , sub_img.shape[1]: ] = (0,0,0)
                sub_img = new_area
            file_name = f'{i}-{j}.jpg' # Formato linha-coluna para identificar futuramente quando for remontar a imagem
            cv2.imwrite(output_path+'/'+file_name, sub_img)
    
    print('\nDivisao concluida\nOs arquivos possuem a nomeclatura: linha-coluna.jpg\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Processa um ortomosaico dividindo em sub imagens JPG.")

    parser.add_argument('--input', required=True, help='Caminho para o arquivo de ortomosaico')
    parser.add_argument('--output', required=True, help='Caminho para o diretorio de saida')
    
    args = parser.parse_args()
    
    cortar_imagem(args.input, args.output)



