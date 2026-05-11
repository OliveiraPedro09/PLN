# Tokenização e Análise de Sentimentos em NLP

## Introdução

A tokenização é uma das etapas fundamentais do NLP (*Natural Language Processing* — Processamento de Linguagem Natural).  
Ela consiste em dividir textos em partes menores chamadas **tokens**, permitindo que algoritmos consigam interpretar linguagem humana.

Esses tokens podem representar:

- palavras;
- frases;
- símbolos;
- números;
- ou até partes de palavras.

A tokenização é amplamente utilizada em sistemas de:

- análise de sentimentos;
- chatbots;
- mecanismos de busca;
- tradutores automáticos;
- classificação de textos;
- inteligência artificial generativa.

---

# O que é Tokenização?

Tokenização é o processo de quebrar um texto em unidades menores chamadas **tokens**.

Exemplo:

```text
"O carro é muito confortável e econômico."
```

Tokens:

```text
["O", "carro", "é", "muito", "confortável", "e", "econômico"]
```

---

# Diferença entre Tokenização de Palavras e Frases

## 1. Tokenização de Palavras

Na tokenização de palavras, o texto é dividido em palavras individuais.

### Exemplo

Texto:

```text
"A camisa é muito confortável."
```

Resultado:

```text
["A", "camisa", "é", "muito", "confortável"]
```

### Objetivo

Esse tipo de tokenização é utilizado para:

- análise de sentimentos;
- contagem de palavras;
- classificação de textos;
- treinamento de modelos de IA;
- identificação de palavras-chave.

---

## 2. Tokenização de Frases

Na tokenização de frases, o texto é dividido em sentenças completas.

### Exemplo

Texto:

```text
"A camisa é confortável. O tecido possui ótima qualidade."
```

Resultado:

```text
[
  "A camisa é confortável.",
  "O tecido possui ótima qualidade."
]
```

### Objetivo

Esse tipo de tokenização é utilizado para:

- separar ideias;
- resumir textos;
- análise contextual;
- tradução automática;
- compreensão semântica.

---

# Exemplos de Tokens em Análise de Sentimentos

A análise de sentimentos utiliza tokens para identificar opiniões positivas, negativas ou neutras em textos.

---

# Loja de Vestuário

## Opinião Negativa

### Tokens de palavras

```text
ruim
desconfortável
rasgado
caro
péssimo
```

### Tokens de frases

```text
"A qualidade da camisa é muito ruim."
"O tecido rasgou após a primeira lavagem."
"O produto não vale o preço."
```

---

## Opinião Positiva

### Tokens de palavras

```text
confortável
bonito
elegante
excelente
resistente
```

### Tokens de frases

```text
"A camisa é muito confortável."
"O tecido possui ótima qualidade."
"A roupa ficou perfeita no corpo."
```

---

## Opinião Neutra

### Tokens de palavras

```text
simples
básico
comum
padrão
```

### Tokens de frases

```text
"A camiseta possui cor azul."
"A calça é tamanho 42."
"O casaco possui dois bolsos."
```

---

# Concessionária Automotiva

## Opinião Negativa

### Tokens de palavras

```text
defeituoso
caro
barulhento
problemático
```

### Tokens de frases

```text
"O carro apresentou muitos defeitos."
"O consumo de combustível é muito alto."
"O veículo faz muito barulho."
```

---

## Opinião Positiva

### Tokens de palavras

```text
econômico
confortável
potente
seguro
moderno
```

### Tokens de frases

```text
"O carro é muito econômico."
"O veículo possui excelente desempenho."
"O automóvel é confortável para viagens."
```

---

## Opinião Neutra

### Tokens de palavras

```text
prata
automático
sedan
flex
```

### Tokens de frases

```text
"O carro possui câmbio automático."
"O veículo é da cor prata."
"O automóvel é modelo sedan."
```

---

# Como a Tokenização é Usada em NLP

A tokenização permite que modelos computacionais transformem linguagem humana em estruturas manipuláveis matematicamente.

Fluxo simplificado:

```text
Texto Original
       ↓
Tokenização
       ↓
Transformação em Dados
       ↓
Processamento por IA
       ↓
Resultado
```

---

# Exemplo Prático de Análise de Sentimentos

Texto:

```text
"O carro é confortável e econômico."
```

Tokens:

```text
["carro", "confortável", "econômico"]
```

Interpretação do modelo:

- confortável → positivo
- econômico → positivo

Resultado:

```text
Sentimento positivo
```

---

# Aplicações Reais

A tokenização é utilizada em:

| Aplicação | Uso |
|---|---|
| Chatbots | Interpretar mensagens |
| E-commerce | Avaliar comentários de produtos |
| Redes sociais | Detectar opiniões |
| Tradutores | Separar sentenças |
| IA Generativa | Processar prompts |
| Motores de busca | Indexar conteúdos |

---

# Conclusão

A tokenização é uma etapa essencial do NLP, permitindo que sistemas computacionais interpretem linguagem natural de forma estruturada.

Existem diferentes tipos de tokenização, sendo as principais:

- tokenização de palavras;
- tokenização de frases.

Essas técnicas são amplamente utilizadas em análise de sentimentos, inteligência artificial, classificação de textos e diversos sistemas modernos baseados em linguagem natural.
