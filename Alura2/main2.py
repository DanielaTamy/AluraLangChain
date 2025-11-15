from langchain.tools import BaseTool
import os
from langchain_openai import AzureChatOpenAI
from pydantic import Field, BaseModel
from typing import ClassVar
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class ExtratorDeEstudante(BaseModel):
    estudante:str = Field("Nome do estudante informado, sempre em letras minúsculas. Exemplo: joão, carlos, joana, carla.")
class DadosDeEstudante(BaseTool):
    name = "DadosDeEstudante"
    description = """Esta ferramenta extrai o histórico e preferências de um estudante de acordo com seu histórico"""
    
    parser: ClassVar= JsonOutputParser(pydantic_object=ExtratorDeEstudante)

    def _run(self, input: str) -> str:
        llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
        )
        parser = JsonOutputParser(pydantic_object=ExtratorDeEstudante)
        template = PromptTemplate(template="""Você deve analisar a {input} e extrar o nome de usuário informado.
                       Formato de saída: 
                       {formato_saida}""",
                       input_variables=["input"],
                       partial_variables={
                           "formato_saida": parser.get_format_instructions("Nome do estudante em letras minúsculas. Ex: joão, carlos, joana, carla.")
                       })
        cadeia = template | llm | parser
        resposta = cadeia.invoke({"input": input})
        print(resposta)
        return resposta['estudante']

pergunta = "Quais os dados da Ana?"
resposta = DadosDeEstudante().run(pergunta)
print(resposta)
