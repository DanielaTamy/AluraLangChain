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
    cidade:str = Field("A cidade recomendada para visitar ")
    motivo:str = Field("O motivo pelo qual é interessante visitar essa cidade")

class Restaurante(BaseModel):
    cidade:str = Field("A cidade recomendada para visitar")
    restaurante:str = Field("O restaurante recomendado na cidade")

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
    template="""Sugira restaurantes populares entre locais em {cidade}
    {formato_de_saida}
    """,
    partial_variables={"formato_de_saida": parseador_restaurante.get_format_instructions()}
)

prompt_cultural= PromptTemplate(
    template="Sugira atividades e locais culturais em {cidade}."
)

cadeia_1 = prompt_cidade | llm | parseador_destino
cadeia_2 = prompt_restaurante | llm | parseador_restaurante
cadeia_3 = prompt_cultural | llm | StrOutputParser()

cadeia = (cadeia_1 | cadeia_2 | cadeia_3)

# Aqui você pode usar invoke diretamente com o prompt
resposta = cadeia.invoke({
    "interesse": "praias"
})

print(resposta)

