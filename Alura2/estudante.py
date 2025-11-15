# estudante.py
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


def busca_dados_de_estudante(nome: str):
    """Busca um estudante no CSV (coluna: USUARIO)."""

    dados = pd.read_csv("documentos/estudantes.csv")
    dados_filtrados = dados[dados["USUARIO"].str.lower() == nome.lower()]

    if dados_filtrados.empty:
        return {}

    return dados_filtrados.to_dict(orient="records")[0]

class ExtratorDeEstudante(BaseModel):
    estudante:str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla.")

class DadosDeEstudante(BaseTool):
    """Ferramenta para extrair o nome de um estudante e buscar no CSV."""

    name : str = "DadosDeEstudante"
    description : str = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico."""

    def _run(self, input: str) -> str:
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
            verbose=True  
            )
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)

        prompt = ChatPromptTemplate.from_template(
            """
            Analise o texto abaixo e extraia o nome do estudante.
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

        estudante = resposta["estudante"].lower()

        dados = busca_dados_de_estudante(estudante)

        return json.dumps(dados, ensure_ascii=False)
        
class ExtratorPerfilAcademico(BaseModel):
    input: str = Field(description="Dados completos do estudante para gerar o perfil acadêmico.")

class Nota(BaseModel):
    area:str = Field("Nome da área de conhecimento")
    nota:float = Field("Nota na área de conhecimento")
    
class PerfilAcademicoDeEstudante(BaseModel):
    name:str = Field("nome do estudante")
    ano_de_conclusao:int = Field("ano de conclusão")
    notas:List[Nota] = Field("Lista de notas das disciplinas e áreas de conhecimento")
    resumo:str = Field("Resumo das principais características desse estudante de forma a torná-lo único e um ótimo potencial estudante para faculdades. Exemplo: só este estudante tem bla bla bla")    

class PerfilAcademico(BaseTool):
    name: str = "PerfilAcademico"
    description: str = """Cria um perfil acadêmico de um estudante. Esta ferramenta requer como entrada todos os dados do estudante. Eu sou incapaz de buscar os dados do estudantes.
    Você tem que buscar os dados do estudante antes de me"""
    args_schema: type[BaseModel] = ExtratorPerfilAcademico

    
    def _run(self, input: str) -> str:

        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
            verbose=True
        )

        parser = JsonOutputParser(pydantic_object=PerfilAcademicoDeEstudante)

        prompt = ChatPromptTemplate.from_template(
            """
            Gere um perfil acadêmico detalhado para o estudante abaixo.

            Diretrizes:
            - Estruture notas, áreas fortes, aptidões e interesses.
            - Sugira universidades e cursos alinhados ao perfil.
            - Produza um resumo final convincente.
            - Estilo: consultora de carreira experiente, objetiva, clara.

            Dados do estudante:
            {dados_do_estudante}

            Formato de saída:
            {formato_de_saida}
            """
        )

        chain = prompt | llm | parser

        resposta = chain.invoke({
            "dados_do_estudante": input,
            "formato_de_saida": parser.get_format_instructions()
        })

        return resposta
