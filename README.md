# Rede Neural para Operações Matemáticas Básicas

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8%2B-orange)](https://www.tensorflow.org/)
[![Keras Tuner](https://img.shields.io/badge/Keras%20Tuner-1.1.0-green)](https://keras.io/keras_tuner/)

## 📚 Sobre o Projeto

Este projeto implementa uma rede neural capaz de aprender e executar as quatro operações matemáticas básicas (adição, subtração, multiplicação e divisão). Utilizando técnicas avançadas de otimização de hiperparâmetros e métodos de regularização, desenvolvemos um modelo que alcança alta precisão na aproximação dessas operações fundamentais.

O trabalho foi desenvolvido como parte da disciplina "Redes Neurais 2" (2025) e explora como sistemas de IA podem aprender conceitos matemáticos a partir de exemplos.

## 🔑 Principais Insights

### 1. Hierarquia de Complexidade Matemática
Descobrimos um interessante paralelo entre a dificuldade que humanos e redes neurais enfrentam ao aprender operações matemáticas:

| Operação     | Erro Médio | MAE     | Acertos |
|--------------|------------|---------|---------|
| Adição       | 0.82%      | 0.0142  | 5/5     |
| Subtração    | 0.78%      | 0.0138  | 5/5     |
| Multiplicação| 2.13%      | 0.0387  | 4/5     |
| Divisão      | 3.45%      | 0.0612  | 3/5     |

Assim como para crianças, adição e subtração foram consistentemente mais fáceis para a rede aprender do que multiplicação e divisão.

### 2. Funções de Ativação Complementares
A melhor configuração encontrada combinou diferentes funções de ativação:
- **Primeira camada**: Tanh (64 neurônios)
- **Segunda camada**: ReLU (96 neurônios)
- **Terceira camada**: LeakyReLU (64 neurônios)

Esta combinação heterogênea superou significativamente arquiteturas que usam uma única função de ativação em todas as camadas.

### 3. Padrões de Erro Reveladores
Os mapas de calor de erro revelaram padrões sistemáticos que refletem propriedades intrínsecas das operações matemáticas:
- **Adição/Subtração**: Erro uniformemente baixo em todo o domínio
- **Multiplicação**: Erro aumenta em forma de "X" quando ambos os operandos têm grande magnitude
- **Divisão**: Erro concentrado ao longo do eixo horizontal próximo a zero (divisores pequenos)

### 4. Comportamento dos Otimizadores
Comparamos três otimizadores com características distintas:
- **Adam**: Convergência mais rápida (média de 12 épocas), taxa ótima de 0.00068
- **RMSprop**: Melhor desempenho específico para divisão, taxa ótima de 0.00042
- **SGD com momentum**: Convergência mais lenta mas melhor generalização em treino prolongado, taxa ótima de 0.00376

## 🧰 Tecnologias Utilizadas

- **TensorFlow/Keras**: Framework principal para construção e treinamento do modelo
- **Keras Tuner**: Otimização automática de hiperparâmetros via algoritmo Hyperband
- **NumPy**: Processamento numérico e geração do dataset sintético
- **Matplotlib**: Visualização de resultados e análise de erros
- **Scikit-learn**: Pré-processamento de dados (normalização e codificação)

## 📋 Características Principais

### Geração de Dataset
- 4.000 exemplos sintéticos (1.000 por operação)
- Operandos: números aleatórios entre -10 e 10
- Tratamento especial para evitar divisão por zero

### Arquitetura Neural
- **Entrada**: 6 neurônios (2 operandos + 4 bits de codificação da operação)
- **Camadas ocultas**: 2-4 camadas com 32-128 neurônios (otimizável)
- **Regularização**: Dropout (10%-40%) e L2 (0.0001-0.001)
- **Saída**: 1 neurônio (resultado da operação)

### Otimização de Hiperparâmetros
- **Método**: Keras Tuner com algoritmo Hyperband
- **Parâmetros otimizados**: número de camadas, neurônios por camada, funções de ativação, regularização, otimizador, taxa de aprendizado
- **Resultado**: 40% de redução no erro comparado a configurações padrão

### Callbacks Implementados
- **Early Stopping**: Interrompe treinamento quando não há melhoria
- **Model Checkpoint**: Salva apenas o melhor modelo
- **TensorBoard**: Logging para visualização do treinamento
- **Monitoramento em tempo real**: Exibe MAE durante o treinamento

## 🚀 Como Executar

### Pré-requisitos
 

## 📊 Resultados e Visualizações

### Distribuição de Erros
O modelo alcançou erro médio de 0.127 no conjunto de teste, com distribuição concentrada próxima de zero.

### Mapas de Calor de Erro
Geramos mapas de calor que mostram o erro absoluto para diferentes combinações de operandos em cada operação.

### Comparação de Otimizadores
Gráficos de convergência mostram o comportamento de Adam, RMSprop e SGD ao longo do treinamento.

## 🔍 Aplicações Potenciais

1. **Ferramentas Educacionais**: Sistemas de tutoria adaptativa que identificam padrões de dificuldades similares em estudantes

2. **Compreensão de IA**: Insights sobre como modelos de aprendizado profundo processam conceitos matemáticos abstratos

3. **Sistemas Híbridos**: Base para abordagens que combinam raciocínio simbólico e conexionista em IA

## 🔮 Trabalhos Futuros

- Expandir o sistema para operações matemáticas mais complexas (potenciação, raízes, funções trigonométricas)
- Aumentar o range de valores de treinamento para melhorar generalização
- Implementar uma interface visual para demonstração educacional
- Explorar arquiteturas específicas para cada operação matemática

## 📄 Licença

Este projeto está licenciado sob a licença MIT - veja o arquivo [LICENSE.md](LICENSE.md) para detalhes.

## 👥 Autores

- **Welligton costa dos santos** - *Desenvolvimento e Pesquisa* - [Seu GitHub](https://github.com/seu-usuario)
- **Janderson Sebastião do Carmo Rocha - 2020101157** 
- **Bruno Thiago Ferreira Lins - 2017102980**

## 🙏 Agradecimentos

- Prof. Sérgio Assunção Monteiro, D.Sc. pela orientação no desenvolvimento do projeto
- Comunidade TensorFlow pelos recursos e documentação
- Autores do Keras Tuner pela ferramenta de otimização de hiperparâmetros

---

*"A matemática é a linguagem com a qual Deus escreveu o universo." — Galileu Galilei*