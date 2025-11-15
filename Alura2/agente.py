# agente.py
from langchain_openai import AzureChatOpenAI
from estudante import DadosDeEstudante, PerfilAcademico
from universidade import DadosDeUniversidade, TodasUniversidades
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("AZURE_OPENAI_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

class AgenteOpenAIFunctions:
    def __init__(self):
        
        self.llm = AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            azure_deployment="gpt-4o-mini",
            api_version="2024-08-01-preview",
            verbose=True,
        )

        # instância da ferramenta
        self.dados_de_estudante = DadosDeEstudante()
        self.perfil_academico = PerfilAcademico()
        self.dados_da_universidade = DadosDeUniversidade()
        self.todas_universidades = TodasUniversidades()

        # registra a ferramenta no modelo
        self.llm_com_tools = self.llm.bind_tools([self.dados_de_estudante, self.perfil_academico, self.dados_da_universidade, self.todas_universidades])
