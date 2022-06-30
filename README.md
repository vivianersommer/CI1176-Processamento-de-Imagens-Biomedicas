# CI1176-Trabalho

O arquivo cintilografias.zip possui imagens anonimizadas de doze pacientes. Os pacientes estão divididos em duas classes: BMT e Grave. Desenvolva um algoritmo de visão computacional que classifique a qual classe um determinado paciente pertence. Utilize a estratégia "Leave-one-patient-out" para fazer os testes, essa técnica consiste em treinar o seu modelo com o número de pacientes - 1  e testar no paciente que ficou de fora. Neste caso serão 12 testes, calcule as métricas vistas em aula e apresente a média e o desvio padrão das mesmas. Lembre-se que nossa base é pequena e que a utilização de Redes Neurais Convolucionais não é adequado neste caso. Elabore um relatório, em formato de artigo, com no máximo quatro páginas descrevendo o que foi desenvolvido e os resultados alcançados. Entregue o código fonte, arquivos de compilação/uso e o relatório em um arquivo único compactado.

# Antes de executar
1. Criar venv.
2. Baixar dependências:
```
pip install -r requirements.txt
```

# Executar
1. Rode o comando, usando a venv
```
python main.py
```

# Alunas
Luzia Millena Santos Silva - GRR20185174 \
Viviane da Rosa Sommer - GRR20182564
