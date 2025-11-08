
# 📚 Módulo 3 - Conversa com Chat

## AULA 1 – Simulando uma Interação de Chat sem Memória

Nesta aula, aprendemos como simular uma conversa com um modelo de linguagem da Azure OpenAI utilizando LangChain, sem o uso de memória de contexto. Cada pergunta é enviada de forma independente, e o modelo responde sem lembrar das interações anteriores.

---

### ✅ Instalação

Certifique-se de instalar os pacotes necessários:

```bash
pip install langchain-openai
pip install python-dotenv
```

---

### 🧠 Objetivo

- Simular uma conversa com múltiplas perguntas.
- Demonstrar que, sem memória, o modelo não mantém o contexto entre as mensagens.

---

### 🧪 Código da Aula

```python
import os 
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

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

lista_perguntas = [
    "Quero visitar um lugar do Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for uma_pergunta in lista_perguntas:
    resposta = llm.invoke(uma_pergunta)
    print(f"Usuário: {uma_pergunta}")
    print(f"IA: {resposta.content}
")
```

---

### 🔍 Observações

- Cada chamada ao `llm.invoke()` é independente.
- O modelo não tem memória entre as interações.
- Para manter o contexto entre mensagens, será necessário adicionar memória nas próximas aulas.

---

### 🔐 Configuração do `.env`

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```


## AULA 2 – Simulando uma Conversa com Memória

Nesta aula, aprendemos como simular uma conversa com memória utilizando LangChain. Ao contrário da aula anterior, agora o modelo consegue lembrar das interações anteriores dentro de uma mesma sessão.

---

### ✅ Objetivo

- Utilizar `InMemoryChatMessageHistory` para manter o histórico da conversa.
- Criar uma cadeia com memória usando `RunnableWithMessageHistory`.
- Simular uma conversa contínua com o modelo AzureChatOpenAI.

---

### 🧠 Componentes Utilizados

- `AzureChatOpenAI`: modelo de linguagem da Azure via LangChain.
- `ChatPromptTemplate`: estrutura de prompt com histórico e entrada do usuário.
- `InMemoryChatMessageHistory`: armazena o histórico da conversa em memória.
- `RunnableWithMessageHistory`: encapsula a cadeia com suporte a sessões de chat.

---

### 🧪 Exemplo de Código

```python
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Configuração do modelo e prompt
prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios."),
    ("placeholder", "{historico}"),
    ("human", "{query}"),
])

cadeia = prompt_sugestao | llm | StrOutputParser()

# Memória por sessão
memoria = {}
def historico_por_sessao(sessao: str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

# Cadeia com memória
cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

# Simulação de conversa
lista_perguntas = [
    "Quero visitar um lugar do Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

for pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke({"query": pergunta}, config={"session_id": "aula_langchain_alura"})
    print(f"Usuário: {pergunta}")
    print(f"IA: {resposta}
")
```

---

### 🔍 Observações

- A memória é mantida por sessão com `session_id`.
- O modelo consegue responder com base nas perguntas anteriores.
- Ideal para simular assistentes conversacionais com contexto.

---

### 🔐 Configuração do `.env`

```env
AZURE_OPENAI_KEY=your_azure_openai_key
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
```
