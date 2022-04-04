# CI1176-Trabalho

#Antes de executar
1. Criar venv.
2. Baixar dependências:
```
pip install -r requirements.txt
```

#Extrutura do código
- main.py:
  - Função principal
- transform_data.py:
  - Recebe um path onde tem imagens DICOM e salva uma imagem jpg para cada uma.
- hog.py
  - Recebe um vetor de imagens jpg e retorna a imagem HOG de cada uma.
    
#Passo a passo
1. Transformação de imagem DICOM em jpg
2. Extração de HOG de cada imagem jpg
3. Treinado modelo MLP com características HOG
4. Avaliação dos modelos com "Leave-one-patient-out"
5. Gerado CSV com média e desvio padrão dos modelos