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

## 🔍 Resumo do Modelo

| Característica | Valor |
|----------------|-------|
| Camadas | 9 |
| Parâmetros Treináveis | 9,377 |
| Otimizador | Adam |
| Função de Perda | MSE |

### Insights Técnicos
- **Funções de Ativação:** ReLU, SELU
- **Total de Camadas Densas:** 4
- **Métrica de Avaliação:** MAE (Erro Absoluto Médio)
- **Arquitetura:** MLP com entrada de 2 valores + codificação one-hot da operação
- **Funções não utilizadas:** tanh, leaky_relu

## 📊 Desempenho por Operação

### Erro Médio por Operação (%)
- **Adição:** 5.9%
- **Subtração:** 41.0%
- **Multiplicação:** 31.4%
- **Divisão:** 155.6%

### Acertos por Operação (≤ 5% erro)
- **Adição:** 4/5 testes (80%)
- **Subtração:** 1/5 testes (20%)
- **Multiplicação:** 2/5 testes (40%)
- **Divisão:** 0/5 testes (0%)

## 💻 Implementação

O projeto utiliza TensorFlow e Keras para implementação da rede neural, juntamente com o Keras Tuner para otimização de hiperparâmetros.

### Principais Componentes:
1. **Geração de Dados:** Dataset sintético com números aleatórios para as quatro operações
2. **Pré-processamento:** Normalização dos valores e codificação one-hot das operações
3. **Arquitetura do Modelo:** Rede neural profunda com camadas densas e regularização
4. **Otimização de Hiperparâmetros:** Utilizando Hyperband para encontrar a melhor configuração
5. **Avaliação:** Testes com exemplos reais e análise detalhada de erros

### Melhores Hiperparâmetros Encontrados:
- **Número de camadas:** 3
- **Otimizador:** Adam
- **Taxa de aprendizado:** 0.001815
- **Camada 1:** 128 neurônios, ReLU, L2=0.0001, Dropout=0.3
- **Camada 2:** 32 neurônios, SELU, L2=0.001, sem Dropout
- **Camada 3:** 128 neurônios, SELU, L2=0.001, Dropout=0.1

## 📈 Resultados e Conclusões

A rede neural conseguiu aprender com maior facilidade operações de adição, enquanto teve dificuldades significativas com divisão. Os resultados mostram que:

- O modelo é excelente para adição (80% de acertos com erro ≤ 5%)
- Razoável para multiplicação (40% de acertos com erro ≤ 5%)
- Limitado para subtração (20% de acertos com erro ≤ 5%)
- Inadequado para divisão (0% de acertos com erro ≤ 5%)

O erro médio absoluto (MAE) final no conjunto de teste foi de aproximadamente 0.0205, indicando um bom desempenho geral, mas com variações significativas entre as operações.

## 🔮 Próximos Passos

Com base nos resultados obtidos, sugerimos as seguintes melhorias:

1. **Modelos Especializados:** Treinar redes separadas para cada operação
2. **Ampliação do Dataset:** Aumentar a quantidade e diversidade dos dados de treinamento
3. **Arquiteturas Alternativas:** Testar RNNs ou Transformers para capturar padrões sequenciais
4. **Processamento Adicional:** Melhorar a normalização de dados para operações de divisão
5. **Técnicas de Ensemble:** Combinar múltiplos modelos para melhorar a precisão geral

## 🚀 Como Executar

```bash
# Instalação das dependências
pip install tensorflow numpy matplotlib sklearn keras-tuner

# Executar o código principal
final.py
```

## 📚 Referências

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
- TensorFlow Documentation: https://www.tensorflow.org/
- Keras Tuner: https://keras.io/keras_tuner/
- Chollet, F. (2021). Deep Learning with Python. Manning Publications.