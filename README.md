# 🧠 Calculadora Neural: Rede Neural para Operações Matemáticas

## 📋 Sobre o Projeto

Este projeto implementa uma rede neural profunda capaz de aprender as quatro operações matemáticas básicas: adição, subtração, multiplicação e divisão. Utilizando técnicas avançadas de deep learning e otimização de hiperparâmetros, conseguimos criar um modelo capaz de realizar cálculos com diferentes níveis de precisão dependendo da operação.

**Autores:**
- Wellington Costa dos Santos - 2019101307
- Janderson Sebastião do Carmo Rocha - 2020101157
- Bruno Thiago Ferreira Lins - 2017102980

**Data:** 10/05/2025  
**Disciplina:** Redes Neurais 2  
**Professor:** Sérgio Assunção Monteiro, D.Sc.

## 1️⃣ Preparação e Validação dos Dados

### Dataset Sintético
- Geramos um dataset contendo 4.000 exemplos (1.000 por operação: adição, subtração, multiplicação e divisão)
- Utilizamos números aleatórios no intervalo [-10, 10], incluindo valores decimais
- Implementamos tratamento especial para evitar divisão por zero (valores menores que 0.01 são substituídos)

### Divisão dos Dados
- **Treino (60%):** 2.400 exemplos para aprendizado do modelo
- **Validação (20%):** 800 exemplos para otimização de hiperparâmetros
- **Teste (20%):** 800 exemplos para avaliação final

### Justificativa da Divisão
Esta proporção foi escolhida para:
1. Garantir dados suficientes para treinamento adequado
2. Manter um conjunto de validação robusto para otimização de hiperparâmetros
3. Reservar uma quantidade representativa para teste independente

### Pré-processamento
- **Normalização:** Utilizamos MinMaxScaler com range=(-1, 1) para operandos e resultados
- **Codificação:** Transformamos os códigos de operação (0-3) em vetores one-hot
- Essa normalização é crucial para equilibrar a influência dos valores e melhorar a convergência

## 2️⃣ Arquitetura da Rede Neural

### Estrutura Inicial
- Implementamos uma MLP com múltiplas camadas densas 
- Exploramos diferentes configurações com 2-4 camadas ocultas
- Finalizamos com uma camada de saída com 1 neurônio (resultado da operação)

### Técnicas de Regularização
- **Dropout:** Aplicado em taxas variáveis (0.1-0.4) para prevenir overfitting
- **Regularização L2:** Implementada com coeficientes 0.001 e 0.0001
- A combinação dessas técnicas provou ser eficaz para melhorar a generalização

### Comparação de Funções de Ativação
Testamos diferentes funções de ativação:
- **ReLU:** Boa performance, especialmente nas primeiras camadas
- **LeakyReLU:** Desempenho similar ao ReLU em nossos testes
- **Tanh:** Não mostrou vantagens significativas para este problema
- **SELU:** Apresentou excelentes resultados nas camadas intermediárias e finais

A combinação vencedora utilizou ReLU na primeira camada e SELU nas subsequentes.

## 3️⃣ Otimização de Hiperparâmetros

### Metodologia
Utilizamos Keras Tuner com o algoritmo Hyperband para busca eficiente, explorando:
- **Número de neurônios:** 32, 64, 96 ou 128 por camada
- **Taxa de aprendizado:** Range de 1e-4 a 1e-2 (escala logarítmica)
- **Coeficientes de regularização L2:** 0.001 ou 0.0001
- **Otimizadores:** Adam, RMSprop e SGD com momentum

### Resultados da Otimização
Após 90 trials (8m12s de processamento):
- **Melhor configuração (Trial #87):** MAE de validação = 0.01958
- **Último trial (#90):** MAE de validação = 0.0805 (significativamente pior)
- **Arquitetura vencedora:** 3 camadas com estrutura "ampulheta" (128→32→128)

### Comparação de Otimizadores
- **Adam:** Mostrou convergência mais rápida e estável (escolhido com taxa de 0.001815)
- **RMSprop:** Performance similar ao Adam, mas ligeiramente menos estável
- **SGD com momentum:** Convergência mais lenta, mas capaz de encontrar bons mínimos

## 4️⃣ Implementação de Callbacks

### Callbacks Utilizados
- **Early Stopping:** Interrompe o treinamento após 5 épocas sem melhoria no MAE de validação
- **ModelCheckpoint:** Salva apenas o melhor modelo baseado no MAE de validação
- **TensorBoard:** Registra métricas para visualização gráfica do treinamento

### Callback Personalizado
Implementamos um LimitadorDeTrials que:
- Controla o número máximo de trials durante a otimização (limite: 59)
- Interrompe trials que excedem este limite para otimizar o tempo total

Adicionalmente, utilizamos um LambdaCallback para exibir métricas em tempo real:
```python
tf.keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoca, logs: print(
        f'Época {epoca+1} - MAE: {logs["mae"]:.4f}, Val MAE: {logs["val_mae"]:.4f}'
    )
)
```

## 5️⃣ Treinamento e Avaliação

### Resultados do Treinamento
- **Métricas no conjunto de teste:** MSE=0.05897, MAE=0.02049
- **Total de parâmetros:** 9.377 (todos treináveis)

### Resumo do Modelo Final

| Característica | Valor |
|----------------|-------|
| Camadas | 9 |
| Parâmetros Treináveis | 9,377 |
| Otimizador | Adam (lr=0.001815) |
| Função de Perda | MSE |

### Desempenho por Operação

| Operação | Erro Médio | Erro Mediano | Erro Máximo | Acertos (≤5% erro) |
|----------|------------|--------------|-------------|----------------|
| Adição | 0.634 | 0.327 | 9.611 | 4/5 (80%) |
| Subtração | 0.946 | 0.738 | 10.407 | 1/5 (20%) |
| Multiplicação | 2.944 | 2.048 | 18.944 | 2/5 (40%) |
| Divisão | 7.408 | 1.064 | 930.638 | 0/5 (0%) |

### Análise de Casos Específicos
Exemplos representativos do conjunto de teste:

**Adição (bom desempenho):**
```
-3.58 + -5.19 = -8.7784 (Predito: -8.9009, Erro: 0.1225)
```

**Subtração (desempenho variável):**
```
-9.90 - 9.67 = -19.5680 (Predito: -26.3897, Erro: 6.8216)
```

**Multiplicação (desafios em valores maiores):**
```
-9.00 * -6.32 = 56.8851 (Predito: 54.2390, Erro: 2.6461)
```

**Divisão (problemas significativos):**
```
-2.11 / -0.89 = 2.3601 (Predito: 0.0550, Erro: 2.3051)
```

### Análise de Overfitting/Underfitting
- Não observamos overfitting significativo graças às técnicas de regularização
- A divergência entre erros por operação sugere que um único modelo pode não ser ideal para todas as operações

## 🔍 Conclusões e Recomendações

### Principais Insights
1. Hierarquia clara de dificuldade: Adição < Subtração < Multiplicação < Divisão
2. A estrutura "ampulheta" (128→32→128) mostrou-se eficiente para capturar padrões matemáticos
3. A combinação ReLU + SELU superou configurações homogêneas de ativação

### Melhorias Propostas
1. **Modelos Especializados:** Treinar redes separadas para cada operação
2. **Pré-processamento Adaptativo:** Diferentes estratégias de normalização por operação
3. **Dataset Expandido:** Maior cobertura de casos extremos, especialmente para divisão
4. **Arquiteturas Alternativas:** Explorar redes mais profundas para operações complexas

## 🚀 Como Executar

```bash
# Instalação das dependências
pip install tensorflow numpy matplotlib sklearn keras-tuner

# Executar o código principal
python final.py
```

## 📚 Referências

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Tuner: https://keras.io/keras_tuner/
- Chollet, F. (2021). Deep Learning with Python. Manning Publications.