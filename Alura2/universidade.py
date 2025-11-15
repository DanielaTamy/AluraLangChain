# universidade.py
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from typing import List

import pandas as pd
import os
from dotenv import load_dotenv
import json
load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class ExtratorDeUniversidade(BaseModel):
    universidade:str = Field("Nome da universidade informado, sempre em letras minúsculas. Exemplo: unesp, ufabc, usp.")


def busca_dados_de_universidade(universidade: str):
    """Busca uma universidade no CSV (coluna: NOME_FACULDADE)."""

    dados = pd.read_csv("documentos/universidades.csv")
    dados["NOME_FACULDADE"] = dados["NOME_FACULDADE"].str.lower()
    dados_filtrados = dados[dados["NOME_FACULDADE"] == universidade.lower()]

    if dados_filtrados.empty:
        return {}

    return dados_filtrados.to_dict(orient="records")[0]

def busca_dados_das_universidades():
    """Busca todas as universidades no CSV."""

    dados = pd.read_csv("documentos/universidades.csv")

    return dados.to_dict(orient="records")

class TodasUniversidades(BaseTool):
    name : str ="TodasUniversidades"
    description : str = """Carrega os dados de todas as universidades. Não é necessário nenhum parâmetro de entrada."""

    def _run(self, input:str):
        universidades = busca_dados_das_universidades()
        return universidades
    
class DadosDeUniversidade(BaseTool):
    name : str = "DadosDeUniversidade"
    description : str = """Esta ferramenta extrai os dados de uma universidade.
Passe para essa ferramenta como argumento o nome da universidade."""

    def _run(self, input: str) -> str:
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
            verbose=True  
            )
        parser = JsonOutputParser(pydantic_object=ExtratorDeUniversidade)

        prompt = ChatPromptTemplate.from_template(
            """
            Analise o texto abaixo e extraia o nome da universidade.
            Retorne no formato JSON exigido.

            Texto: {input}

            Formato de saída:
            {formato_saida}
            """
        )

        chain = prompt | llm | parser

        resposta = chain.invoke({
            "input": input,
            "formato_saida": parser.get_format_instructions()
        })

        universidade = resposta["universidade"].lower()

        dados = busca_dados_de_universidade(universidade)

        return json.dumps(dados, ensure_ascii=False)