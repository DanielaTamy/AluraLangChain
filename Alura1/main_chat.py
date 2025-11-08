import os 
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

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

prompt_sugestao = ChatPromptTemplate.from_messages([
    ("system", "Você é um guia de viagem especializado em destinos brasileiros. Apresente-se como Sr. Passeios."),
    ("placeholder", "{historico}"),
    ("human", "{query}"),
])

cadeia = prompt_sugestao | llm | StrOutputParser()

memoria = {}
sessao = "aula_langchain_alura"

def historico_por_sessao(sessao : str):
    if sessao not in memoria:
        memoria[sessao] = InMemoryChatMessageHistory()
    return memoria[sessao]

lista_perguntas = [
    "Quero visitar um lugar do Brasil, famoso por praias e cultura. Pode sugerir?",
    "Qual a melhor época do ano para ir?"
]

cadeia_com_memoria = RunnableWithMessageHistory(
    runnable=cadeia,
    get_session_history=historico_por_sessao,
    input_messages_key="query",
    history_messages_key="historico"
)

for uma_pergunta in lista_perguntas:
    resposta = cadeia_com_memoria.invoke({"query": uma_pergunta}, config={"session_id": sessao})
    print(f"Usuário: {uma_pergunta}")
    print(f"IA: {resposta}\n")
