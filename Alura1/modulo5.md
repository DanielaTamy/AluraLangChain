
# Module 5 - Implementing RAG (Retrieval-Augmented Generation)

# aula 1
This module demonstrates how to implement a Retrieval-Augmented Generation (RAG) pipeline using LangChain with two different vector store backends: FAISS and Chroma.

---

## 🔧 Environment Setup

Install the required packages:
```bash
python -m pip install langchain-community
pip install faiss-cpu
pip install langchain-openai
pip install langchain-text-splitter
```

> ⚠️ **Note:** FAISS is not compatible with Python 3.13. If you're using Python 3.13, consider using **Chroma** instead.

---

## 🧠 Using FAISS (Not compatible with Python 3.13)
```python
from dotenv import load_dotenv
from openai import api_version, embeddings
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-08-01-preview"
DEPLOYMENT_EMBEDDINGS = "text-embedding-3-large"

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version=api_version,
    temperature=0.5
)

embeddings = AzureOpenAIEmbeddings(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment=DEPLOYMENT_EMBEDDINGS,
    api_version=api_version
)

documento = TextLoader("documentos/GTB_gold_Nov23.txt", encoding="utf-8").load()

pedacos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documento)

dados_recuperados = FAISS.from_documents(pedacos, embeddings).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido."),
    ("human", "{query}

Contexto: 
{context}

Resposta:")
])

cadeia = prompt | llm | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "

".join(t.page_content for t in trechos)
    return cadeia.invoke({"query": pergunta, "context": contexto})

print(responder("Como devo proceder caso tenha um item roubado?"))
```

---

## 🧠 Using Chroma (Recommended for Python 3.13)
```python
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-08-01-preview"
DEPLOYMENT_EMBEDDINGS = "text-embedding-3-large"

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version=api_version,
    temperature=0.5
)

embeddings = AzureOpenAIEmbeddings(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment=DEPLOYMENT_EMBEDDINGS,
    api_version=api_version
)

documento = TextLoader("documentos/GTB_gold_Nov23.txt", encoding="utf-8").load()

pedacos = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documento)

dados_recuperados = Chroma.from_documents(pedacos, embeddings).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido."),
    ("human", "{query}

Contexto: 
{context}

Resposta:")
])

cadeia = prompt | llm | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "

".join(t.page_content for t in trechos)
    return cadeia.invoke({"query": pergunta, "context": contexto})

print(responder("Como devo proceder caso tenha um item roubado?"))
```

---

## ✅ Observações
- Use FAISS apenas se estiver em uma versão compatível do Python (até 3.12).
- Chroma é uma alternativa moderna e compatível com Python 3.13.
- Certifique-se de configurar corretamente as variáveis de ambiente `.env` com sua chave e endpoint do Azure OpenAI.

---

## 📁 Arquivo de Documento
Certifique-se de que o arquivo `GTB_gold_Nov23.txt` esteja presente na pasta `documentos/` para que o carregamento funcione corretamente.

---

## 🧪 Teste
Exemplo de pergunta:
```python
print(responder("Como devo proceder caso tenha um item roubado?"))
```

Essa chamada irá recuperar os trechos relevantes do documento e gerar uma resposta com base no conteúdo.

## Aula 2 - Integração de PDFs com LangChain

Nesta aula, você aprenderá como importar e processar múltiplos arquivos PDF usando LangChain, com o carregador `PyPDFLoader` e o vetor store `Chroma`, compatível com Python 3.13.

---

## 🛠 Instalação de Dependências

Instale os pacotes necessários:
```bash
C:/Users/849770/AppData/Local/Programs/Python/Python313/python.exe -m pip install pypdf
pip install langchain-community
pip install langchain-openai
pip install langchain-text-splitter
```

---

## 💻 Código de Implementação
```python
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-08-01-preview"
DEPLOYMENT_EMBEDDINGS = "text-embedding-3-large"

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version=api_version,
    temperature=0.5
)

embeddings = AzureOpenAIEmbeddings(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment=DEPLOYMENT_EMBEDDINGS,
    api_version=api_version
)

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf"
]

documentos = sum([
    PyPDFLoader(arquivo).load() for arquivo in arquivos
], [])

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
).split_documents(documentos)

dados_recuperados = Chroma.from_documents(
    pedacos,
    embeddings
).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido."),
    ("human", "{query}

Contexto: 
{context}

Resposta:")
])

cadeia = prompt | llm | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "

".join(t.page_content for t in trechos)
    return cadeia.invoke({"query": pergunta, "context": contexto})

print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold"))
```

---

## ✅ Observações
- O carregador `PyPDFLoader` permite importar conteúdo de arquivos PDF.
- O vetor store `Chroma` é compatível com Python 3.13 e substitui FAISS.
- O modelo de linguagem usado é `gpt-4o-mini` via Azure OpenAI.
- O sistema responde com base exclusivamente no conteúdo dos documentos PDF.

---

## 🧪 Exemplo de Pergunta
```python
print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold"))
```

Essa chamada irá recuperar os trechos relevantes dos documentos e gerar uma resposta contextualizada.