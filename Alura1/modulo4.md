
# 🧭 Módulo 4 - Orquestração

## AULA 1 – Criando a base de uma solução com LCEL e LangGraph

Nesta aula, iniciamos a construção de uma solução orquestrada utilizando LangChain Expression Language (LCEL) e LangGraph. O objetivo é estruturar uma cadeia de execução simples com um modelo de linguagem da Azure OpenAI.

---

### ✅ Objetivo

- Criar um assistente de viagens com base em um prompt estruturado.
- Utilizar LCEL para compor a cadeia de execução.

---

### 📦 Bibliotecas utilizadas

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
```

---

### ⚙️ Configuração do modelo

```python
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not endpoint:
    raise ValueError("A chave da API ou o endpoint não foram definidos no .env")

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version="2024-05-01-preview",
    temperature=0.5
)

print("Chave e endpoint carregados com sucesso!")
```

---

### 🧠 Criação do Assistente

```python
prompt_consultor = ChatPromptTemplate.from_messages([
    ("system", "Você é um consultor de viagens."),
    ("human", "{query}"),
])

assistente = prompt_consultor | llm | StrOutputParser()

resposta = assistente.invoke({"query": "Quero férias em praias no Brasil."})
print(resposta)
```

---

### 🔍 Observações

- A cadeia `assistente` é composta por três etapas:
  1. `ChatPromptTemplate`: estrutura o prompt com mensagens do sistema e do usuário.
  2. `AzureChatOpenAI`: envia o prompt ao modelo da Azure.
  3. `StrOutputParser`: extrai a resposta como texto simples.

- Essa estrutura é a base para soluções mais complexas com múltiplas etapas e lógica condicional, que serão abordadas nas próximas aulas.

---

### 🔐 Configuração do `.env`

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```


## AULA 2 – Roteamento de Cadeias com LangChain

Nesta aula, aprendemos como criar um sistema de roteamento inteligente que direciona perguntas para cadeias especializadas com base no conteúdo da consulta. Utilizamos o modelo AzureChatOpenAI com suporte a `with_structured_output` para interpretar a intenção do usuário.

---

### ✅ Objetivo

- Criar duas cadeias especializadas: uma para destinos de praia e outra para montanha.
- Utilizar um roteador para decidir qual cadeia deve ser usada com base na pergunta do usuário.

---

### 📦 Bibliotecas utilizadas

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Literal
import os
```

---

### ⚙️ Configuração do modelo

```python
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
    temperature=0.5
)
```

---

### 🧠 Lógica de Roteamento

```python
class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com 'praia' ou 'montanha'"),
    ("human", "{query}"),
])

roteador = prompt_roteador | llm.with_structured_output(Rota)
```

---

### 🧩 Cadeias especializadas

```python
prompt_consultor_praia = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sra. Praia. Você é uma especialista em destinos para praia."),
    ("human", "{query}"),
])

prompt_consultor_montanha = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sr. Montanha. Você é um especialista em destinos para montanha e atividades radicais."),
    ("human", "{query}"),
])

cadeia_praia = prompt_consultor_praia | llm | StrOutputParser()
cadeia_montanha = prompt_consultor_montanha | llm | StrOutputParser()
```

---

### 🔁 Função de roteamento

```python
def responda(pergunta: str):
    rota = roteador.invoke({"query": pergunta})["destino"]
    print(rota)
    if rota == "praia":
        return cadeia_praia.invoke({"query": pergunta})
    return cadeia_montanha.invoke({"query": pergunta})

print(responda("Quero surfar em um lugar quente."))
```

---

### 🔍 Observações

- O roteador usa `with_structured_output(Rota)` para garantir que a resposta seja um JSON válido.
- A versão da API deve ser `2024-08-01-preview` ou superior para suportar `json_schema`.
- O sistema é facilmente extensível para mais destinos ou critérios de roteamento.

---

### 🔐 Configuração do `.env`

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```


## Aula 3 – Orquestrando Assistentes com LangGraph

## 📦 Instalação de dependências

```bash
python -m pip install langgraph langchain-openai python-dotenv
```

## 🧠 Objetivo da Aula

Aprender a **orquestrar múltiplos assistentes especializados** usando o LangGraph, criando um fluxo de decisão que direciona a consulta do usuário para o assistente mais adequado (praia ou montanha).

---

## 🔐 Carregando variáveis de ambiente

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not endpoint:
    raise ValueError("A chave da API ou o endpoint não foram definidos no .env")
```

---

## 🤖 Configurando o modelo LLM

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
    temperature=0.5
)
```

---

## 🧩 Criando os assistentes especializados

### Consultor de Praia

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt_consultor_praia = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sra. Praia. Você é uma especialista em destinos para praia."),
    ("human", "{query}"),
])

cadeia_praia = prompt_consultor_praia | llm | StrOutputParser()
```

### Consultor de Montanha

```python
prompt_consultor_montanha = ChatPromptTemplate.from_messages([
    ("system", "Apresente-se como Sr. Montanha. Você é um especialista em destinos para montanha e atividades radicais."),
    ("human", "{query}"),
])

cadeia_montanha = prompt_consultor_montanha | llm | StrOutputParser()
```

---

## 🧭 Criando o roteador de destinos

```python
from typing import TypedDict, Literal

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]

prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com 'praia' ou 'montanha'"),
    ("human", "{query}"),
])

roteador = prompt_roteador | llm.with_structured_output(Rota)
```

---

## 🔄 Funções dos nós do grafo

```python
from langchain_core.runnables import RunnableConfig

class Estado(TypedDict):
    query: str
    destino: Rota
    resposta: str

async def no_roteador(estado: Estado, config=RunnableConfig):
    return {"destino": await roteador.ainvoke({"query": estado["query"]}, config=config)}

async def no_praia(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_praia.ainvoke({"query": estado["query"]}, config)}

async def no_montanha(estado: Estado, config=RunnableConfig):
    return {"resposta": await cadeia_montanha.ainvoke({"query": estado["query"]}, config)}
```

---

## 🧠 Lógica de decisão

```python
def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"
```

---

## 🕸️ Montando o grafo com LangGraph

```python
from langgraph.graph import StateGraph, START, END

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()
```

---

## 🚀 Executando o fluxo

```python
import asyncio

async def main():
    resposta = await app.ainvoke({"query": "Quero escalar."})
    print(resposta["resposta"])

asyncio.run(main())
```

---

## ✅ Resultado Esperado

O sistema identifica que "escalar" está relacionado a montanha e direciona a consulta para o **Sr. Montanha**, que responde com sugestões de destinos ou atividades radicais.
