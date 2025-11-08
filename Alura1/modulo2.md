
# 📘 MÓDULO 2
## AULA 1 – Cadeia de execução com LangChain

Nesta aula, aprendemos como criar uma cadeia de execução utilizando `PromptTemplate`, `AzureChatOpenAI` e `StrOutputParser` para sugerir cidades com base em interesses específicos.

---

### ✅ Instalação

Certifique-se de instalar os pacotes necessários:

```bash
pip install langchain-openai
pip install python-dotenv
```

---

### 🧠 Código da Aula

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Verifica se as variáveis foram carregadas corretamente
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

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dados o meu interesse por {interesse}
    """,
    input_variables=["interesse"],
)

cadeia = prompt_cidade | llm | StrOutputParser()

# Aqui você pode usar invoke diretamente com o prompt
resposta = cadeia.invoke({
    "interesse": "praias"
})

print(resposta)
```

---

### 🔍 Explicações

- **PromptTemplate**: Define um modelo de prompt com variáveis dinâmicas.
- **AzureChatOpenAI**: Conecta ao modelo GPT-4o via Azure OpenAI.
- **StrOutputParser**: Extrai a resposta como uma string simples.
- **Cadeia**: Conecta os componentes em uma sequência lógica de execução.

---

### 📌 Observações

- Certifique-se de que o nome do deployment (`gpt-4o-mini`) esteja corretamente configurado no portal do Azure.
- As variáveis de ambiente devem estar definidas no arquivo `.env`:

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```

## AULA 2 – Respostas Estruturadas com Pydantic e JSONOutputParser

Nesta aula, aprendemos como utilizar `JsonOutputParser` da LangChain em conjunto com `Pydantic` para gerar respostas estruturadas em formato JSON, facilitando o processamento e validação dos dados retornados pelo modelo.

---

### ✅ Instalação

Certifique-se de instalar os pacotes necessários:

```bash
pip install langchain-openai
pip install python-dotenv
```

---

### 🧠 Código da Aula

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.globals import set_debug
import os

set_debug(True)

load_dotenv()

# Verifica se as variáveis foram carregadas corretamente
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

class Destino(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    motivo: str = Field("O motivo pelo qual é interessante visitar essa cidade")

parseador = JsonOutputParser(pydantic_object=Destino)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dados o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador.get_format_instructions()}
)

cadeia = prompt_cidade | llm | parseador

# Aqui você pode usar invoke diretamente com o prompt
resposta = cadeia.invoke({
    "interesse": "praias"
})

print(resposta)
```

---

### 🔍 Explicações

- **Pydantic**: Define um modelo de dados com validação automática.
- **JsonOutputParser**: Garante que a resposta do modelo esteja no formato JSON esperado.
- **PromptTemplate**: Inclui instruções de formatação no prompt.
- **cadeia**: Conecta o prompt, o modelo e o parser em uma sequência lógica.

---

### 📌 Observações

- Certifique-se de que o nome do deployment (`gpt-4o-mini`) esteja corretamente configurado no portal do Azure.
- As variáveis de ambiente devem estar definidas no arquivo `.env`:

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```

## AULA 3 – Criando uma sequência de cadeias com LCEL

Nesta aula, aprendemos como criar uma sequência de execução usando LCEL (LangChain Expression Language), conectando múltiplas etapas de processamento com LangChain.

---

### ✅ Instalação

Certifique-se de instalar os pacotes necessários:

```bash
pip install langchain-openai
pip install python-dotenv
```

---

### 🧠 Objetivo

Criar uma cadeia de execução que:
1. Sugere uma cidade com base em um interesse.
2. Sugere um restaurante popular nessa cidade.
3. Sugere atividades culturais na mesma cidade.

---

### 🧩 Componentes Utilizados

- `PromptTemplate`: para estruturar os prompts.
- `AzureChatOpenAI`: para gerar respostas usando GPT-4o via Azure.
- `JsonOutputParser`: para garantir que a resposta esteja em formato JSON.
- `StrOutputParser`: para respostas livres em texto.
- `Pydantic`: para definir modelos de dados estruturados.

---

### 🧪 Código da Aula

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_core.globals import set_debug
import os

set_debug(True)
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

class Destino(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    motivo: str = Field("O motivo pelo qual é interessante visitar essa cidade")

class Restaurante(BaseModel):
    cidade: str = Field("A cidade recomendada para visitar")
    restaurante: str = Field("O restaurante recomendado na cidade")

parseador_destino = JsonOutputParser(pydantic_object=Destino)
parseador_restaurante = JsonOutputParser(pydantic_object=Restaurante)

prompt_cidade = PromptTemplate(
    template="""
    Sugira uma cidade dados o meu interesse por {interesse}.
    {formato_de_saida}
    """,
    input_variables=["interesse"],
    partial_variables={"formato_de_saida": parseador_destino.get_format_instructions()}
)

prompt_restaurante = PromptTemplate(
    template="""
    Sugira restaurantes populares entre locais em {cidade}
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurante.get_format_instructions()}
)

prompt_cultural = PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}."
)

cadeia_1 = prompt_cidade | llm | parseador_destino
cadeia_2 = prompt_restaurante | llm | parseador_restaurante
cadeia_3 = prompt_cultural | llm | StrOutputParser()

cadeia = cadeia_1 | cadeia_2 | cadeia_3

resposta = cadeia.invoke({"interesse": "praias"})
print(resposta)
```

---

### 📌 Observações

- O uso de `|` permite encadear etapas de forma declarativa.
- Cada etapa recebe a saída da anterior como entrada.
- É importante garantir que os formatos de entrada e saída estejam compatíveis.

---

### 🛠️ Requisitos do `.env`

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```
