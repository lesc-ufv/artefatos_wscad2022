# Artefatos WSCAD 2022
## Título: HPyC-FPGA - Integração de Aceleradores em FPGA de Alto Desempenho com Python para Jupyter Notebooks
### Resumo:
O desenvolvimento de aceleradores em FPGAs (Field Programmable Gates Arrays) ainda é um desafio. Recentemente, o ambiente PYNQ da Xilinx possibilitou a integração de código Python com aceleradores em FPGA. A maioria dos exemplos está voltada para placas de prototipação utilizadas no desenvolvimento de aplicações embarcadas. Este artigo apresenta o algoritmo K-means de aprendizado de máquina não supervisionado como estudo de caso. A principal contribuição deste trabalho é o encapsulamento de 3 aceleradores acoplados com PYNQ usando o ambiente Jupyter Notebook. A avaliação foi realizada em uma máquina de alto desempenho utilizando um FPGA Alveo U55C com memória HBM (High Bandwidth Memory). Os resultados são promissores, além de mostrar as facilidades de uso do FPGA de forma encapsulada, o ganho de desempenho foi de uma a duas ordens de grandezas em comparação a um sistema com dois processadores Xeon(R) Silver 4210R com 10 núcleos cada, executando a etapa de classificação do algoritmo K-means.

### K-means RTL

Os aceleradores k-means em RTL foram criados por meio do gerador disponivel em [https://github.com/lesc-ufv/kmeans_aws_f1_hw_generator](https://github.com/lesc-ufv/kmeans_aws_f1_hw_generator)

### K-means HLS

Os aceleradores k-means em HLS estão disponiveis no conjunto de benchmark Rodinia HLS disponíveis em [https://github.com/SFU-HiAccel/rodinia-hls/tree/master/Benchmarks/kmeans](https://github.com/SFU-HiAccel/rodinia-hls/tree/master/Benchmarks/kmeans)

Observação: Foram utlizados dois projetos que estão presentes, sendo o baseline e o mais otimizado que usa multiplos bancos de memoria. Para realizar a compilação da versão com multiplos bancos de memoria foi necessário editar o [Makefile](https://github.com/SFU-HiAccel/rodinia-hls/blob/master/Benchmarks/kmeans/kmeans_6_multiddr/Makefile) nas linhas 34,35, e 36 substituindo o módulos de DDR por HBM para placas FPGAs que possuem memórias HBM. 
