# 🧠 Calculadora Neural: Rede Neural para Operações Matemáticas

## 📋 Índice
- [Sobre o Projeto](#-sobre-o-projeto)
- [Resumo Executivo](#-resumo-executivo)
- [Preparação e Validação dos Dados](#-preparação-e-validação-dos-dados)
- [Arquitetura da Rede Neural](#-arquitetura-da-rede-neural)
- [Otimização de Hiperparâmetros](#-otimização-de-hiperparâmetros)
- [Implementação de Callbacks](#-implementação-de-callbacks)
- [Treinamento e Avaliação](#-treinamento-e-avaliação)
- [Conclusões e Recomendações](#-conclusões-e-recomendações)
- [Limitações e Desafios](#-limitações-e-desafios)
- [Trabalhos Futuros](#-trabalhos-futuros)
- [Como Executar](#-como-executar)
- [Uso Prático do Modelo](#-uso-prático-do-modelo)
- [Referências](#-referências)

## 📋 Sobre o Projeto

Este projeto implementa uma rede neural profunda capaz de aprender as quatro operações matemáticas básicas: adição, subtração, multiplicação e divisão. Utilizando técnicas avançadas de deep learning e otimização de hiperparâmetros, conseguimos criar um modelo capaz de realizar cálculos com diferentes níveis de precisão dependendo da operação.

**Autores:**
- Wellington Costa dos Santos - 2019101307
- Janderson Sebastião do Carmo Rocha - 2020101157
- Bruno Thiago Ferreira Lins - 2017102980

**Data:** 10/05/2025  
**Disciplina:** Redes Neurais 2  
**Professor:** Sérgio Assunção Monteiro, D.Sc.

## 📊 Resumo Executivo

Nossa pesquisa implementou uma rede neural capaz de aprender as quatro operações matemáticas básicas, utilizando uma arquitetura MLP com configuração "ampulheta" (128→32→128 neurônios). Principais resultados:

- **Performance Global:** MSE=0.05897, MAE=0.02049 no conjunto de teste
- **Diferenças por Operação:** Adição (melhor desempenho) > Subtração > Multiplicação > Divisão (pior desempenho)
- **Otimização:** Utilizamos Keras Tuner com Hyperband para encontrar a melhor configuração entre 90 tentativas
- **Melhor Modelo:** Combinação de ReLU na primeira camada e SELU nas subsequentes, com regularização L2 e dropout

Esta abordagem demonstra o potencial das redes neurais para modelar operações matemáticas, mas também revela limitações importantes para operações mais complexas como divisão.

## 1️⃣ Preparação e Validação dos Dados

### Dataset Sintético
- Geramos um dataset contendo 4.000 exemplos (1.000 por operação: adição, subtração, multiplicação e divisão)
- Utilizamos números aleatórios no intervalo [-10, 10], incluindo valores decimais
- Implementamos tratamento especial para evitar divisão por zero (valores menores que 0.01 são substituídos)

```python
def gerar_dataset_operacoes(amostras_por_operacao=1000):
    """
    Gera um dataset sintético para treinar operações matemáticas.
    """
    # Cálculo do número total de operações
    total_amostras = amostras_por_operacao * 4  # 4 operações: +, -, *, /
    
    # Geração de números aleatórios entre -10 e 10
    operando_1 = np.random.uniform(-10, 10, total_amostras)
    operando_2 = np.random.uniform(-10, 10, total_amostras)
    
    # Prevenção de divisão por zero: substitui valores próximos de zero
    indice_inicio_divisao = amostras_por_operacao * 3
    operando_2[indice_inicio_divisao:][np.abs(operando_2[indice_inicio_divisao:]) < 0.01] = 0.01
    
    # Criação dos códigos de operação (0: soma, 1: subtração, 2: multiplicação, 3: divisão)
    codigos_operacao = np.concatenate([
        np.zeros(amostras_por_operacao),          # Adição (0)
        np.ones(amostras_por_operacao),           # Subtração (1)
        np.full(amostras_por_operacao, 2),        # Multiplicação (2)
        np.full(amostras_por_operacao, 3)         # Divisão (3)
    ])
    
    # Combinação dos operandos e códigos de operação como entrada (X)
    X = np.column_stack((operando_1, operando_2, codigos_operacao))
    
    # Cálculo dos resultados esperados (y)
    resultados = np.concatenate([
        # Adição: a + b
        operando_1[:amostras_por_operacao] + operando_2[:amostras_por_operacao],
        
        # Subtração: a - b
        operando_1[amostras_por_operacao:2*amostras_por_operacao] - 
        operando_2[amostras_por_operacao:2*amostras_por_operacao],
        
        # Multiplicação: a * b
        operando_1[2*amostras_por_operacao:3*amostras_por_operacao] * 
        operando_2[2*amostras_por_operacao:3*amostras_por_operacao],
        
        # Divisão: a / b
        operando_1[3*amostras_por_operacao:] / operando_2[3*amostras_por_operacao:]
    ])
    
    return X, resultados
```

### Divisão dos Dados
- **Treino (60%):** 2.400 exemplos para aprendizado do modelo
- **Validação (20%):** 800 exemplos para otimização de hiperparâmetros
- **Teste (20%):** 800 exemplos para avaliação final

```python
# Divisão dos dados em conjuntos de treino (60%), validação (20%) e teste (20%)
X_temp, X_teste, y_temp, y_teste = train_test_split(
    X_completo, y_completo, test_size=0.2, random_state=SEED
)
X_treino, X_validacao, y_treino, y_validacao = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=SEED
)
```

### Justificativa da Divisão
Esta proporção foi escolhida para:
1. Garantir dados suficientes para treinamento adequado
2. Manter um conjunto de validação robusto para otimização de hiperparâmetros
3. Reservar uma quantidade representativa para teste independente

### Pré-processamento
- **Normalização:** Utilizamos MinMaxScaler com range=(-1, 1) para operandos e resultados
- **Codificação:** Transformamos os códigos de operação (0-3) em vetores one-hot
- Essa normalização é crucial para equilibrar a influência dos valores e melhorar a convergência

```python
# Inicialização dos transformadores para pré-processamento
escala_entrada = MinMaxScaler(feature_range=(-1, 1))  # Normaliza números para [-1, 1]
escala_saida = MinMaxScaler(feature_range=(-1, 1))    # Normaliza resultados para [-1, 1]
codificador_operacoes = OneHotEncoder(sparse_output=False)  # Transforma códigos em vetores one-hot

def transformar_dados(X, y, ajustar_transformadores=False):
    """
    Normaliza os valores numéricos e codifica as operações.
    """
    # Separação dos operandos e dos códigos de operação
    numeros = X[:, :2]                 # Primeiras duas colunas (operandos)
    operacoes = X[:, 2].reshape(-1, 1)  # Terceira coluna (códigos de operação)
    
    # Normalização e codificação
    if ajustar_transformadores:
        # Primeira vez: ajusta os transformadores aos dados
        numeros_norm = escala_entrada.fit_transform(numeros)
        y_norm = escala_saida.fit_transform(y.reshape(-1, 1)).flatten()
        operacoes_codificadas = codificador_operacoes.fit_transform(operacoes)
    else:
        # Usa transformadores já ajustados
        numeros_norm = escala_entrada.transform(numeros)
        y_norm = escala_saida.transform(y.reshape(-1, 1)).flatten()
        operacoes_codificadas = codificador_operacoes.transform(operacoes)
    
    # Combinação dos dados processados
    return np.hstack([numeros_norm, operacoes_codificadas]), y_norm
```

## 2️⃣ Arquitetura da Rede Neural

### Estrutura Inicial
- Implementamos uma MLP com múltiplas camadas densas 
- Exploramos diferentes configurações com 2-4 camadas ocultas
- Finalizamos com uma camada de saída com 1 neurônio (resultado da operação)

```python
def construir_modelo(hp):
    """
    Função para construir o modelo com hiperparâmetros otimizáveis.
    """
    # Inicialização do modelo sequencial
    modelo = tf.keras.Sequential()
    
    # Camada de entrada com forma definida pelos dados processados
    modelo.add(tf.keras.layers.Input(shape=(X_treino_norm.shape[1],)))
    
    # Número de camadas ocultas a ser otimizado (entre 2 e 4)
    num_camadas = hp.Int("num_camadas", min_value=2, max_value=4, step=1)
    
    # Adição das camadas ocultas com hiperparâmetros otimizáveis
    for i in range(num_camadas):
        # Número de neurônios na camada
        unidades = hp.Int(f"unidades_{i}", min_value=32, max_value=128, step=32)
        
        # Regularização L2 para prevenção de overfitting
        regularizador = tf.keras.regularizers.l2(
            hp.Choice(f"l2_{i}", values=[0.001, 0.0001])
        )
        
        # Adição da camada densa
        modelo.add(tf.keras.layers.Dense(
            units=unidades,
            kernel_regularizer=regularizador
        ))
        
        # Função de ativação
        ativacao = hp.Choice(
            f"ativacao_{i}", 
            values=["relu", "tanh", "selu", "leaky_relu"]
        )
        
        if ativacao == "leaky_relu":
            modelo.add(tf.keras.layers.LeakyReLU())
        else:
            modelo.add(tf.keras.layers.Activation(ativacao))
        
        # Dropout opcional para regularização adicional
        if hp.Boolean(f"usar_dropout_{i}"):
            taxa_dropout = hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.4, step=0.1)
            modelo.add(tf.keras.layers.Dropout(rate=taxa_dropout))
    
    # Camada de saída (uma única unidade para o resultado da operação)
    modelo.add(tf.keras.layers.Dense(1))
    
    # ... [continuação do código de compilação]
    
    return modelo
```

### Diagrama da Arquitetura Final

```
Entrada (6 unidades: 2 operandos + 4 one-hot)
    ↓
Camada Densa (128 unidades, ReLU, L2=0.0001)
    ↓
Dropout (30%)
    ↓
Camada Densa (32 unidades, SELU, L2=0.001)
    ↓
Camada Densa (128 unidades, SELU, L2=0.001)
    ↓
Dropout (10%)
    ↓
Camada de Saída (1 unidade)
```

### Técnicas de Regularização
- **Dropout:** Aplicado em taxas variáveis (0.1-0.4) para prevenir overfitting
- **Regularização L2:** Implementada com coeficientes 0.001 e 0.0001
- A combinação dessas técnicas provou ser eficaz para melhorar a generalização

```python
# Trecho destacando a implementação de regularização
regularizador = tf.keras.regularizers.l2(
    hp.Choice(f"l2_{i}", values=[0.001, 0.0001])
)

# Adição da camada densa com regularização
modelo.add(tf.keras.layers.Dense(
    units=unidades,
    kernel_regularizer=regularizador
))

# Implementação de Dropout
if hp.Boolean(f"usar_dropout_{i}"):
    taxa_dropout = hp.Float(f"dropout_{i}", min_value=0.1, max_value=0.4, step=0.1)
    modelo.add(tf.keras.layers.Dropout(rate=taxa_dropout))
```

### Comparação de Funções de Ativação
Testamos diferentes funções de ativação:
- **ReLU:** Boa performance, especialmente nas primeiras camadas
- **LeakyReLU:** Desempenho similar ao ReLU em nossos testes
- **Tanh:** Não mostrou vantagens significativas para este problema
- **SELU:** Apresentou excelentes resultados nas camadas intermediárias e finais

```python
# Código que implementa a seleção de diferentes funções de ativação
ativacao = hp.Choice(
    f"ativacao_{i}", 
    values=["relu", "tanh", "selu", "leaky_relu"]
)

if ativacao == "leaky_relu":
    modelo.add(tf.keras.layers.LeakyReLU())
else:
    modelo.add(tf.keras.layers.Activation(ativacao))
```

A combinação vencedora utilizou ReLU na primeira camada e SELU nas subsequentes.

## 3️⃣ Otimização de Hiperparâmetros

### Metodologia
Utilizamos Keras Tuner com o algoritmo Hyperband para busca eficiente, explorando:
- **Número de neurônios:** 32, 64, 96 ou 128 por camada
- **Taxa de aprendizado:** Range de 1e-4 a 1e-2 (escala logarítmica)
- **Coeficientes de regularização L2:** 0.001 ou 0.0001
- **Otimizadores:** Adam, RMSprop e SGD com momentum

```python
# Inicialização do otimizador de hiperparâmetros (Hyperband)
otimizador_hiperparametros = kt.Hyperband(
    construir_modelo,
    objective="val_mae",           # Objetivo: minimizar o erro absoluto médio na validação
    max_epochs=30,                 # Número máximo de épocas para cada trial
    factor=3,                      # Fator de redução para o Hyperband
    directory="diretorio_tuning",  # Diretório para armazenar resultados
    project_name="mlp_matematica"  # Nome do projeto
)

# Seleção do otimizador como um hiperparâmetro
otimizador_nome = hp.Choice(
    "otimizador", 
    values=["adam", "rmsprop", "sgd"]
)

# Taxa de aprendizado otimizável
taxa_aprendizado = hp.Float(
    "taxa_aprendizado", 
    min_value=1e-4, 
    max_value=1e-2, 
    sampling="log"
)

# Criação do otimizador escolhido com a taxa de aprendizado
otimizadores = {
    "adam": tf.keras.optimizers.Adam(learning_rate=taxa_aprendizado),
    "rmsprop": tf.keras.optimizers.RMSprop(learning_rate=taxa_aprendizado),
    "sgd": tf.keras.optimizers.SGD(learning_rate=taxa_aprendizado, momentum=0.9)
}
otimizador = otimizadores[otimizador_nome]
```

### Resultados da Otimização
Após 90 trials (8m12s de processamento):
- **Melhor configuração (Trial #87):** MAE de validação = 0.01958
- **Último trial (#90):** MAE de validação = 0.0805 (significativamente pior)
- **Arquitetura vencedora:** 3 camadas com estrutura "ampulheta" (128→32→128)

```
Trial 90 Complete [00h 00m 13s]
val_mae: 0.08050110936164856
Best val_mae So Far: 0.01958031952381134
Total elapsed time: 00h 08m 12s
```

### Curva de Convergência

A figura abaixo mostra a evolução do MAE de validação para os diferentes trials ao longo do processo de otimização:

```
    MAE
0.08 |    *                             *
     |        *             *              
     |  *         *     *       *          
0.04 |     *  *     *       *             
     |            *   *  *                
     |                      *   *         
0.02 |                          *         
     +--------------------------------
         0   20   40   60   80   Trial#
```

### Comparação de Otimizadores
- **Adam:** Mostrou convergência mais rápida e estável (escolhido com taxa de 0.001815)
- **RMSprop:** Performance similar ao Adam, mas ligeiramente menos estável
- **SGD com momentum:** Convergência mais lenta, mas capaz de encontrar bons mínimos

```python
# Compilação do modelo com o otimizador escolhido
modelo.compile(
    optimizer=otimizador,
    loss='mse',                # Erro quadrático médio para regressão
    metrics=['mae']            # Erro absoluto médio como métrica adicional
)
```

## 4️⃣ Implementação de Callbacks

### Callbacks Utilizados
- **Early Stopping:** Interrompe o treinamento após 5 épocas sem melhoria no MAE de validação
- **ModelCheckpoint:** Salva apenas o melhor modelo baseado no MAE de validação
- **TensorBoard:** Registra métricas para visualização gráfica do treinamento

```python
otimizador_hiperparametros.search(
    X_treino_norm, y_treino_norm,
    validation_data=(X_validacao_norm, y_validacao_norm),
    callbacks=[
        # Early stopping para interromper trials sem progresso
        tf.keras.callbacks.EarlyStopping(
            patience=5,                   # Número de épocas sem melhoria
            restore_best_weights=True     # Restaura os melhores pesos
        ),
        
        # Checkpoint para salvar o melhor modelo
        tf.keras.callbacks.ModelCheckpoint(
            "melhor_modelo.keras",
            save_best_only=True           # Salva apenas quando há melhoria
        ),
        
        # TensorBoard para visualização do treinamento
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
        
        # Outros callbacks...
    ],
    verbose=2  # Nível de detalhamento das mensagens
)
```

### Callback Personalizado
Implementamos um LimitadorDeTrials que:
- Controla o número máximo de trials durante a otimização (limite: 59)
- Interrompe trials que excedem este limite para otimizar o tempo total

```python
# Callback personalizado para limitar o número de trials
class LimitadorDeTrials(tf.keras.callbacks.Callback):
    """
    Callback para limitar o número de trials durante a busca de hiperparâmetros.
    """
    def __init__(self, max_trials=59):
        super().__init__()
        self.max_trials = max_trials
        self.trial_atual = 0
    
    def on_train_begin(self, logs=None):
        self.trial_atual += 1
        if self.trial_atual > self.max_trials:
            print(f"\n⛔ Atingido o limite de {self.max_trials} trials. Interrompendo busca.")
            self.model.stop_training = True
```

Adicionalmente, utilizamos um LambdaCallback para exibir métricas em tempo real:
```python
# Callback personalizado para mostrar progresso em tempo real
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

```python
# Avaliação no conjunto de teste
avaliacao_final = melhor_modelo.evaluate(X_teste_norm, y_teste_norm)
print("\n✅ Avaliação final (MSE, MAE):", avaliacao_final)
# Resultado: [0.058975763618946075, 0.02049693651497364]
```

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

### Gráfico de Comparação de Desempenho por Operação

```
Erro Médio por Operação:
    Erro
 8.0 |                               █
     |                               █
 6.0 |                               █
     |                               █
 4.0 |                               █
     |                      █        █
 2.0 |                      █        █
     |           █          █        █
 0.0 |     █     █          █        █
     +-------------------------------------
           Adição Subtração Multi  Divisão
```

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
- Os resultados indicam que a rede generaliza bem para adição, mas tem dificuldades crescentes com operações mais complexas

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

## 🚧 Limitações e Desafios

Durante o desenvolvimento do projeto, enfrentamos diversos desafios e identificamos limitações importantes:

1. **Precisão da Divisão:** O modelo apresentou dificuldades significativas com a operação de divisão, especialmente com denominadores próximos a zero, refletindo a natureza matematicamente mais complexa desta operação.

2. **Generalização para Valores Extremos:** Os maiores erros ocorreram em casos com valores nos extremos do intervalo [-10, 10], indicando a necessidade de melhor cobertura de exemplos nessas regiões.

3. **Otimização Computacional:** A busca de hiperparâmetros demandou recursos significativos (8m12s para 90 trials), indicando a necessidade de otimização para uso em ambientes com recursos limitados.

4. **Modelo Único vs. Especializado:** Uma única rede para todas as operações apresenta vantagens de implementação, mas compromete a precisão, especialmente nas operações mais complexas.

5. **Range Limitado:** Os resultados sugerem que o modelo atual pode não generalizar bem para valores fora do intervalo de treinamento [-10, 10].

## 🔭 Trabalhos Futuros

Com base nos resultados obtidos e nas limitações identificadas, planejamos as seguintes direções para pesquisas futuras:

1. **Arquiteturas Especializadas:** Desenvolver e comparar modelos dedicados para cada operação matemática, potencialmente aumentando a complexidade para multiplicação e divisão.

2. **Exploração de Técnicas de Atenção:** Implementar mecanismos de atenção para melhorar a capacidade do modelo de focar em diferentes aspectos dos operandos dependendo da operação.

3. **Expansão para Operações Mais Complexas:** Treinar o modelo para realizar operações como exponenciação, raiz quadrada e operações com múltiplos operandos.

4. **Implementação em Dispositivos de Baixo Recurso:** Otimizar o modelo para execução em dispositivos móveis ou embarcados, explorando técnicas de quantização e compressão.

5. **Abordagem Híbrida Neural-Simbólica:** Combinar a rede neural com um sistema de regras para melhorar o desempenho em casos especiais (como divisão por números próximos a zero).

## 🚀 Como Executar

```bash
# Instalação das dependências
pip install tensorflow numpy matplotlib sklearn keras-tuner

# Executar o código principal
python final.py

# Visualizar logs no TensorBoard
tensorboard --logdir=logs_tensorboard/
```

## 🔌 Uso Prático do Modelo

Após o treinamento, você pode utilizar o modelo para realizar operações matemáticas conforme o exemplo abaixo:

```python
# Carregar o modelo treinado
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Carregar modelo e transformadores
modelo = keras.models.load_model("melhor_modelo.keras")
escala_entrada = MinMaxScaler(feature_range=(-1, 1))
escala_saida = MinMaxScaler(feature_range=(-1, 1))
codificador_operacoes = OneHotEncoder(sparse_output=False)

# Carregar os estados dos transformadores (código simplificado)
# Na prática, você precisaria salvar e carregar estes estados

def calcular_operacao(a, b, operacao):
    """
    Realiza o cálculo usando o modelo neural.
    
    Parâmetros:
        a, b: Operandos (valores entre -10 e 10)
        operacao: Código da operação (0: +, 1: -, 2: *, 3: /)
        
    Retorna:
        Resultado da operação
    """
    # Pré-processar operandos
    numeros = np.array([[a, b]])
    numeros_norm = escala_entrada.transform(numeros)
    
    # Pré-processar código da operação
    operacao_reshape = np.array([[operacao]])
    operacao_cod = codificador_operacoes.transform(operacao_reshape)
    
    # Combinar para entrada final
    entrada_processada = np.hstack([numeros_norm, operacao_cod])
    
    # Fazer previsão
    resultado_norm = modelo.predict(entrada_processada, verbose=0)[0, 0]
    resultado = escala_saida.inverse_transform([[resultado_norm]])[0, 0]
    
    return resultado

# Exemplo de uso
a, b = 5.7, -3.2
print(f"{a} + {b} = {calcular_operacao(a, b, 0)}")
print(f"{a} - {b} = {calcular_operacao(a, b, 1)}")
print(f"{a} * {b} = {calcular_operacao(a, b, 2)}")
print(f"{a} / {b} = {calcular_operacao(a, b, 3)}")
```