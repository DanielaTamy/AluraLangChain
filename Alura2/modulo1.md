
# 📘 Aula 1 — Ferramentas (Tools) com LangChain + Azure OpenAI

## ✅ Módulo 1 — Documentação Completa

### 🚀 1. Preparando o Ambiente Virtual

**Criar o ambiente virtual:**
```bash
python -m venv .venv
```

**Ativar o ambiente:**
```bash
.venv\Scripts\activate
```

### 📦 2. Instalando Dependências

**Instalar o requirements.txt (caso exista):**
```bash
pip install -r requirements.txt
```

**Instalar LangChain e integrações Azure/OpenAI:**
```bash
pip install langchain-openai
```

**Forçar instalação correta do LangChain (Python 3.13):**
```bash
C:/Users/849770/AppData/Local/Programs/Python/Python313/python.exe -m pip install langchain
```

**Corrigir versão do pydantic (LangChain depende disso):**
```bash
pip install "pydantic<3"
```

---

### 🧩 3. Estrutura do Código — Visão Geral

Nesta aula construímos uma Tool (ferramenta) personalizada usando:
- ✅ LangChain 0.3+
- ✅ Pydantic 2
- ✅ Azure OpenAI (modelo gpt-4o-mini)
- ✅ JsonOutputParser (gera e valida JSON)
- ✅ ClassVar para evitar conflitos com Pydantic

A ferramenta recebe uma frase:
> “Quais os dados da Ana?”

E extrai apenas o nome do estudante, garantindo sempre letras minúsculas.

---

### 🧱 4. Código Completo da Aula
```python
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
import os
from pydantic import BaseModel, Field
from typing import ClassVar
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class ExtratorDeEstudante(BaseModel):
    estudante: str = Field(
        description="Nome do estudante informado, sempre em letras minúsculas. Ex: joão, carlos, joana, carla."
    )

class DadosDeEstudante(BaseTool):
    name: str = "DadosDeEstudante"
    description: str = (
        "Esta ferramenta extrai o histórico e preferências de um estudante "
        "de acordo com seu histórico."
    )
    
    parser: ClassVar = JsonOutputParser(pydantic_object=ExtratorDeEstudante)

    def _run(self, input: str) -> str:
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
        )
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)

        template = ChatPromptTemplate.from_template(
            "Você deve analisar o texto: {input}
"
            "E extrair os dados no formato: {formato_saida}"
        )

        cadeia = template | llm | parser

        resposta = cadeia.invoke({
            "input": input,
            "formato_saida": parser.get_format_instructions()
        })

        print(resposta)
        return resposta["estudante"]

pergunta = "Quais os dados da Ana?"
resultado = DadosDeEstudante().run(pergunta)
print(resultado)
```

---

### 🧠 5. Explicando o Código — Linha por Linha

#### ✅ 5.1. Carregando ambiente e dependências
```python
load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
```
Carrega as credenciais do serviço Azure OpenAI a partir do arquivo `.env`.

#### ✅ 5.2. Modelo Pydantic (Esquema do JSON)
```python
class ExtratorDeEstudante(BaseModel):
    estudante: str = Field(...)
```
Define como o JSON deve ser estruturado e quais campos são obrigatórios.

#### ✅ 5.3. A Tool — DadosDeEstudante
```python
class DadosDeEstudante(BaseTool):
```
Uma Tool LangChain usada em agentes, grafos e rotinas automatizadas.

#### ✅ Por que usar `ClassVar`?
```python
parser: ClassVar = JsonOutputParser(...)
```
Evita que o Pydantic trate `parser` como campo de dados.

#### ✅ 5.4. Execução da Tool (`_run`)
Inicializa o modelo `gpt-4o-mini` da Azure e monta o pipeline:
```python
template | llm | parser
```

---

### 🔁 6. O que mudou do código antigo para o novo?

| Item                  | Antes         | Agora         | Motivo                              |
|----------------------|---------------|---------------|-------------------------------------|
| Pydantic versão      | v1            | v2            | LangChain atualizado                |
| `pydantic_v1` import | Funcionava    | Não existe    | Removido no LangChain 0.3          |
| Campo `parser`       | Atributo simples | `ClassVar` | Evitar que vire campo do modelo    |
| `JsonOutputParser`   | Retornava BaseModel | Retorna dict | Mudança interna do parser          |
| Prompt               | Sintaxe antiga | `from_template` | Nova API padrão                  |
| Tool                 | Atributos soltos | Validação completa | Nova arquitetura Pydantic     |
| AzureChatOpenAI      | API flexível  | Exige `api_version` | Nova validação interna         |

---

### ✅ 7. Resultado Final

A execução imprime:
```json
{'estudante': 'ana'}
```
E depois:
```
ana
```
Mostrando que:
- O JSON foi gerado corretamente
- O nome foi extraído
- A Tool está funcionando com Azure OpenAI

---

### 🎓 8. Conclusão

Nesta aula você aprendeu:
- ✅ Como preparar o ambiente com LangChain moderno
- ✅ Como criar uma Tool profissional usando Azure OpenAI
- ✅ Como usar JsonOutputParser com Pydantic 2
- ✅ Por que `ClassVar` é obrigatório nesses casos
- ✅ Como montar um pipeline completo (prompt → LLM → parser)
- ✅ Como adaptar código antigo para a nova versão do LangChain



--------------------------------------------------------

## aula 2
pip install langchainhub

pip install "langchain==0.1.20"
pip install "langchain-openai==0.1.6"
pip install "langchain-community==0.0.29"
pip uninstall langchain-core -y
pip install "langchain-core==0.1.52"

pip install "pydantic<3"

