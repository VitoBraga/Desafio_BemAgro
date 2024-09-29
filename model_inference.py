import keras
import cv2
import os
import argparse
import numpy as np
import re



def prepare_img(img_path):
    formato_imagens = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']
    original_imgs = []

    # Verifica se a imagem existe
    if os.path.exists(img_path):

        # Verifica se o caminho é de uma única imagem ou uma pasta com imagens
        if img_path.split('.')[-1] in formato_imagens:

            img = cv2.imread(img_path)
            
            # Verifica necessidade de corrigir o tamanho das imagens - Aplicando mesma divisao do orthomosaico
            if img.shape[0] != 500 or img.shape[1] != 500:
                imgs, batch = ajustar_shape(img)
            else:
                imgs = [img]
                batch = [1,1]
            
            # Faz a preparação das imagens de entrada da rede
            for i in range(len(imgs)):
                original_imgs.append(imgs[i])
                imgs[i] = prepare_x_input(imgs[i])
            
            return np.array(imgs), [batch], np.array(original_imgs)
            

        else:
            # caso seja pasta com imagens trata de forma a incluir todas as imagens (ou as que estiverem no padrão desejado)
            imgs, batch, original_imgs = execucao_em_batch(img_path)

            return np.array(imgs), batch, np.array(original_imgs)

    else:
        print('O caminho da imagem não existe')
        return False

def execucao_em_batch(img_path):
    formato_imagens = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG']
    original_imgs = []
    # Definição do padrão esperado pras imagens em caso de execução em lote (padrão do orthomosaico)
    padrao_ideal = re.compile(r'^\d+-\d+\.(jpg|JPG|png|PNG|jpeg|JPEG|tif|tiff|TIF|TIFF)$')

    imgs_x = os.listdir(img_path)

    arquivos_imagens = 0
    arquivos_imagens_padrao = 0
    padrao = []
    imgs = []
    batchs = []

    # Verificacao do diretório para entender a organização dos arquivos
    # a ideia é contabilizar quantos arquivos estão dentro do formato desejado
    for file in imgs_x:
        if file.split('.')[-1] in formato_imagens:
            arquivos_imagens += 1
            if padrao_ideal.match(file):
                arquivos_imagens_padrao += 1
                padrao.append(file)
    
    # Se mais da metade dos arquivos (apenas imagens) estão no formato desejado, usaremos apenas estes arquivos.
    # Caso contrário usaremos todas as imagens na ordem que estiverem
    batch_padrao = False
    if arquivos_imagens_padrao >= (arquivos_imagens*0.51):
        # Faz a organização dos arquivos no formato linha-coluna para facilitar na junção das imagens no final
        imgs_x =  sorted(padrao, key=lambda x: (int(x.split('-')[0]), int(x.split('-')[1].split('.')[0])))
        linhas = int(imgs_x[-1].split('-')[0]) +1
        colunas = int(imgs_x[-1].split('-')[1].split('.')[0]) +1
        batchs = [[linhas, colunas]]
        batch_padrao = True

    for imagem in imgs_x:
        if imagem.split('.')[-1] in formato_imagens:
            img = cv2.imread(img_path+'/'+imagem)

            # Verifica necessidade de corrigir o tamanho das imagens - Aplicando mesma divisao do orthomosaico
            if img.shape[0] != 500 or img.shape[1] != 500:
                ajustado, batch = ajustar_shape(img)
                if not batch_padrao:
                    batchs.append(batch)

            else:
                ajustado = [img]
                if not batch_padrao:
                    batchs.append([1,1])
            
            for imagem in ajustado:
                original_imgs.append(imagem)
                imagem = prepare_x_input(imagem)
                imgs.append(imagem)
    
    return imgs, batchs, np.array(original_imgs)

def ajustar_shape(img):
    imgs = []

    # Reaproveitando código da divisão de orthomosiacos
    altura_original = img.shape[0]
    comprimento_original = img.shape[1]

    altura = 500
    comprimento = 500
    
    columns = int(comprimento_original / comprimento) +1
    rows = int(altura_original / altura) +1
    
    for i in range(rows):
        
        for j in range(columns):
            
            sub_img = img[i*altura : (i+1)*altura, j*comprimento : (j+1)*comprimento, :]
            
            if sub_img.shape[0] < 500 or sub_img.shape[1] < 500:
                new_area = np.zeros((altura, comprimento, 3), dtype=np.uint8)
                new_area[:sub_img.shape[0], : sub_img.shape[1] ] = sub_img
                new_area[ sub_img.shape[0]: , sub_img.shape[1]: ] = (0,0,0)
                sub_img = new_area

            # Em vez de escrever as imagens como na divisão, vamos agrupalas em uma lista e processar tudo junto
            # a ideia é exibir apenas o resultado final com as images reunidas
            imgs.append(sub_img)
    
    return imgs, [rows, columns]

def prepare_x_input(image):
    # Reaproveitando do treinamento do modelo
    # Realçar características básicas da vegetação para auxiliar na segmentação realizada pela rede
    # Aumento de saturação para realçar as cores, seguido da seleção de canal verde

    fator_saturacao = 10

    imagem_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)    
    imagem_hsv[:, :, 1] = cv2.multiply(imagem_hsv[:, :, 1], fator_saturacao)
    imagem_saturada = cv2.cvtColor(imagem_hsv, cv2.COLOR_HSV2BGR)
    green_filter = imagem_saturada[ : , : , 1]

    green_filter = green_filter/255
    
    return green_filter

def carregar_modelo(modelo_path):
    if os.path.exists(modelo_path):
        modelo = keras.models.load_model(modelo_path)
        return modelo

    else:
        print('Diretorio de modelo nao encontrado')
        return False

def previsao(x, modelo, output_path, batches, original_imgs):
    predict = modelo.predict(x)
    predict = (predict > 0.5).astype(np.uint8) * 255 # Ajusta a saída para ser obrigatoriamente binária

    img_mask = []

    print(len(predict))
    # Aplica a segmentação nas imagens de entrada
    for i in range(len(predict)):
        # Criar uma máscara a partir da imagem de thresholding emitida pela rede
        _, mask = cv2.threshold(predict[i], 200, 255, cv2.THRESH_BINARY)

        cor_sobreposicao = (255, 0, 0)

        # Criar uma imagem com a mesma dimensão da original, com a cor de sobreposição
        sobreposicao = np.zeros_like(original_imgs[i])
        sobreposicao[:] = cor_sobreposicao

        # Usar a máscara para sobrepor a cor sobre as áreas detectadas
        resultado = original_imgs[i].copy()
        resultado[mask == 255] = cv2.addWeighted(original_imgs[i][mask == 255], 0.7, sobreposicao[mask == 255], 0.3, 0)


        img_mask.append(resultado)
    
    # Preparação para unir novamente as imagens
    linhas_gerais = 0
    colunas_gerais = 0

    for imagem in batches:
        linhas = imagem[0]
        colunas = imagem[1]
        linhas_gerais += linhas
        colunas_gerais += colunas
    
    # Criação do escopo da imagem geral
    image_final = np.zeros(((linhas_gerais*500), (colunas_gerais*500), 3), dtype=np.uint8)


    # Preenchendo o escopo com as previsões mescladas a imagem original
    img_counter = 0
    for imagem in batches:

        linhas = imagem[0]
        colunas = imagem[1]

        for i in range(linhas):
            for j in range(colunas):
                linha_inicio = (i * 500) 
                linha_fim = ((i+1)*500)
                coluna_inicio = (j * 500)
                coluna_fim = ((j+1)*500)

                image_final[linha_inicio : linha_fim, coluna_inicio : coluna_fim, :] = img_mask[img_counter]
                img_counter += 1
            
    
    cv2.imwrite(output_path, image_final)
    print('\nConcluido\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Segmentacao com modelo de CNN para vegetacao")

    parser.add_argument('--rgb', required=True, help='Caminho para o diretorio com imagens - input da rede')
    parser.add_argument('--modelpath', required=True, help='Caminho onde o modelo da rede esta')
    parser.add_argument('--output', required=True, help='Caminho para o diretorio onde ira salvar o resultado da segmentacao')

    args = parser.parse_args()

    print('\nIniciando preparo das imagens...')
    imgs, batchs, original_imgs = prepare_img(args.rgb)
    print('\nCarregando modelo...')
    modelo = carregar_modelo(args.modelpath)
    print('\nIniciando previsao')
    previsao(imgs, modelo, args.output, batchs, original_imgs)

