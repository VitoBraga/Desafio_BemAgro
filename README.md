**Feito por:** João Victor da Silva Braga

**Bibliotecas necessárias de instalação:**
1. Numpy (pip install numpy)
2. Keras (pip install keras)
3. OpenCV (pip install opencv-python)

As imagens geradas a partir do mosaico, são dividas em grade, seguindo o padrão "linha-coluna".

A segmentação foi realizada utilizando Thresholding com base na coloração da vegetação, seguido de pós processamento morfológico para suavização do thresholding.

Recomenda-se presença de GPU para treinamento da rede neural.

A inferência pode ser realizada em um arquivo de imagem único, ou conjunto de imagens que sigam o padrão "linha-coluna". Para isto, basta passar o diretório com as imagens. 

OBS: Tamanho mínimo das imagens é de 500 x 500 px
