# MODULO 1
# AULA 1 
# 🎵 Roteiro de Viagem com LangChain e Azure OpenAI

Este é um mini projeto que utiliza a API do **Azure OpenAI** em conjunto com **LangChain** para gerar roteiros de viagem personalizados com base em preferências da família, como número de dias, número de crianças e atividades favoritas (ex: música, natureza, aventura).

## 🚀 Funcionalidades

- Geração de roteiros de viagem personalizados usando **GPT-4o** via Azure OpenAI.
- Configuração de variáveis sensíveis com **dotenv**.
- Estrutura pronta para expansão com **LangChain** e integração com outras ferramentas como FAISS e PDF parsing.

## 📦 Requisitos

Instale as dependências com:

```bash
python -m pip install -r requirements.txt
```

Ou instale manualmente:

```bash
python -m pip install openai
python -m pip install python-dotenv
```

## 📁 Estrutura do Projeto

```
.
├── main.py               # Código principal do projeto
├── .env                  # Arquivo com variáveis de ambiente (não versionado)
├── requirements.txt      # Lista de dependências
└── .gitignore            # Arquivos ignorados pelo Git
```

## 🔐 Configuração

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```

> **Importante:** Nunca compartilhe sua chave de API publicamente.

## 🧠 Como funciona

O script utiliza a biblioteca `openai` para se conectar ao serviço Azure OpenAI. Ele gera um roteiro de viagem com base em parâmetros definidos no código:

```python
numero_dias = 7
numero_criancas = 2
atividade = "música"
```

Esses dados são usados para construir um prompt que é enviado ao modelo **GPT-4o** via Azure.

## 📌 Exemplo de uso

```bash
python main.py
```

Saída esperada:

```
Dia 1: Chegada e visita a um museu de música interativo...
Dia 2: Oficina de instrumentos musicais para crianças...
...
```

## 🛠️ Tecnologias utilizadas

- [Python](https://www.python.org/)
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [PyPDF](https://pypi.org/project/pypdf/)

## 📌 Observações

- Certifique-se de ter uma conta no Azure com acesso ao serviço OpenAI.
- O nome do deployment (`gpt-4o-mini`) deve estar corretamente configurado no portal do Azure.


# AULA 2 – Adicionando um Prompt Template

Nesta aula, vamos aprender como utilizar o `PromptTemplate` da LangChain para estruturar melhor os prompts enviados ao modelo da Azure OpenAI.

---

### ✅ Instalação

Certifique-se de instalar o pacote necessário:

```bash
python -m pip install langchain-openai
```

---

### 🧠 Código Corrigido

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Verifica se as variáveis foram carregadas corretamente
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not endpoint:
    raise ValueError("A chave da API ou o endpoint não foram definidos no .env")

# Inicializa o modelo AzureChatOpenAI
llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version="2024-05-01-preview",
    temperature=0.5
)

print("Chave e endpoint carregados com sucesso!")

# Dados do roteiro
numero_dias = 7
numero_criancas = 2
atividade = "praia"

# Criação do template de prompt
modelo_de_prompt = PromptTemplate.from_template(
    """
    Crie um roteiro de viagem de {dias} dias, 
    para uma família com {numero_criancas} crianças, 
    que busca atividades relacionadas a {atividade}.
    """
)

# Formata o prompt com os dados fornecidos
prompt = modelo_de_prompt.format(
    dias=numero_dias,
    numero_criancas=numero_criancas,
    atividade=atividade
)

print("Prompt gerado:
", prompt)

# Envia o prompt ao modelo
resposta = llm.invoke(prompt)

# Exibe a resposta
print("
Resposta do modelo:
", resposta.content)
```

---

### 📝 Explicações

- **`PromptTemplate.from_template`**: Cria um template de prompt com variáveis que podem ser preenchidas dinamicamente.
- **`llm.invoke(prompt)`**: Envia o prompt diretamente ao modelo e retorna a resposta.
- **`resposta.content`**: A resposta gerada pelo modelo.

---

### 📌 Observações

- Certifique-se de que o nome do deployment (`gpt-4o-mini`) esteja corretamente configurado no portal do Azure.
- As variáveis de ambiente devem estar definidas no arquivo `.env`:

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```




