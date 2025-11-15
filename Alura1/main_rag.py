
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()
api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_version = "2024-08-01-preview"

DEPLOYMENT_EMBEDDINGS = "text-embedding-3-large"  # coloque o NOME DO DEPLOYMENT aqui (não o modelo)

llm = AzureChatOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment="gpt-4o-mini",
    api_version=api_version,
    temperature=0.5
)

embeddings = AzureOpenAIEmbeddings(
    api_key=api_key,
    azure_endpoint=endpoint,
    azure_deployment=DEPLOYMENT_EMBEDDINGS,
    api_version=api_version
)

arquivos = [
    "documentos/GTB_standard_Nov23.pdf",
    "documentos/GTB_gold_Nov23.pdf",
    "documentos/GTB_platinum_Nov23.pdf"
]

documentos = sum(
    [PyPDFLoader(arquivo).load() for arquivo in arquivos],
    []
)

pedacos = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
).split_documents(documentos)

# ✅ Usando Chroma no lugar de FAISS
dados_recuperados = Chroma.from_documents(
    pedacos,
    embeddings
).as_retriever(search_kwargs={"k": 2})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Responda usando exclusivamente o conteúdo fornecido."),
    ("human", "{query}\n\nContexto: \n{context}\n\nResposta:"),
])

cadeia = prompt | llm | StrOutputParser()

def responder(pergunta: str):
    trechos = dados_recuperados.invoke(pergunta)
    contexto = "\n\n".join(t.page_content for t in trechos)
    return cadeia.invoke({"query": pergunta, "context": contexto})

print(responder("Como devo proceder caso tenha um item comprado roubado e caso eu tenha o cartão gold"))
