from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
import asyncio
import os 

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not api_key or not endpoint:
    raise ValueError("A chave da API ou o endpoint não foram definidos no .env")

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version="2024-08-01-preview",
    temperature=0.5
)

print("Chave e endpoint carregados com sucesso!")


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

class Rota(TypedDict):
    destino: Literal["praia", "montanha"]


prompt_roteador = ChatPromptTemplate.from_messages([
    ("system", "Responda apenas com 'praia' ou 'montanha'"),
    ("human", "{query}"),
])

roteador = prompt_roteador | llm.with_structured_output(Rota)

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

def escolher_no(estado: Estado) -> Literal["praia", "montanha"]:
    return "praia" if estado["destino"]["destino"] == "praia" else "montanha"

grafo = StateGraph(Estado)
grafo.add_node("rotear", no_roteador)
grafo.add_node("praia", no_praia)
grafo.add_node("montanha", no_montanha)

grafo.add_edge(START, "rotear")
grafo.add_conditional_edges("rotear", escolher_no)
grafo.add_edge("praia", END)
grafo.add_edge("montanha", END)

app = grafo.compile()

async def main():
    resposta = await app.ainvoke({"query": "Quero escalar."})
    print(resposta["resposta"])


asyncio.run(main())