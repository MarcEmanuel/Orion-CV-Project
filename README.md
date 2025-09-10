# Orion-CV-Project
A computer vision challenge from the Orion study group, focused on polyp detection and segmentation using classic feature extraction and classification models on the Kvasir-SEG dataset.


#Visão Geral
Este projeto é um desafio de visão computacional realizado por mim e por Arthur no grupo de estudos Orion. Nosso objetivo era aplicar algoritmos e modelos clássicos de machine learning para a segmentação e detecção de pólipos no dataset Kvasir-SEG. O desafio nos permitiu aprofundar o conhecimento sobre as bases da visão computacional, focando em técnicas que foram amplamente utilizadas antes do avanço dos modelos de deep learning.

#Metodologia
Inicialmente nos utilizamos as mascaras nas imagens e extraimos features de comprimento, intensidade e de textura, para a de textura usamos os filtros de Haralick.
Após isso tiramos o uso da mascara e tetamos outros filtros e metodos como: HOG, GABOR, LBP, tentamos a de varias formas.

#Classificação: 
Os descritores extraídos foram usados como entrada para modelos de classificação, utilizamos tanto o SVM quanto o Radom Florest e comparamos ambos em todos os códigos.

#Contribuidores
Este projeto foi realizado em colaboração com:

Marcos Emanuel de Sales Pereira - https://github.com/MarcEmanuel

Arthur Rodrigues - https://github.com/arthurrsampaio
