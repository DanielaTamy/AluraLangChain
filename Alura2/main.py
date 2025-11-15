from agente import AgenteOpenAIFunctions

agente = AgenteOpenAIFunctions()

query = """
Você é um assistente inteligente com acesso a duas ferramentas:

1️⃣ DadosDeEstudante — busca dados de um estudante no CSV.
2️⃣ PerfilAcademico — cria um perfil acadêmico a partir dos dados do estudante.

Sua tarefa é decidir, de forma autônoma, qual ferramenta usar (ou nenhuma)
para responder à pergunta do usuário.

Pergunta: "Dentre todas as faculdades disponíveis, quais Ana tem mais chance de entrar?"

"""

resposta = agente.llm_com_tools.invoke(query, tool_choice="auto")

print("🧠 Resposta bruta do modelo:", resposta)

tool_calls = getattr(resposta, "tool_calls", [])

resultados = []

if not tool_calls:
    print("❌ Nenhuma ferramenta foi chamada.")
else:
    print(f"🔧 {len(tool_calls)} ferramenta(s) chamada(s):")

    for call in tool_calls:
        nome_tool = call.get("name")
        args = call.get("args", {})

        print(f"\n📌 Tool Call detectado:")
        print(f"   • Tool: {nome_tool}")
        print(f"   • Args: {args}")

        if nome_tool == "DadosDeEstudante":
            entrada = args.get("input", "")
            resultado = agente.dados_de_estudante.run(entrada)
            resultados.append(resultado)
            print("✅ Resultado da ferramenta:", resultado)

        elif nome_tool == "PerfilAcademico":
            entrada = args.get("input", "")
            resultado = agente.perfil_academico.run(entrada)
            resultados.append(resultado)
            print("✅ Resultado da ferramenta:", resultado)
        
        
        elif nome_tool == "DadosDeUniversidade":
            entrada = args.get("input", "")
            resultado = agente.dados_da_universidade.run(entrada)
            resultados.append(resultado)
            print("✅ Resultado da ferramenta:", resultado)

    print("\n📚 TODOS OS RESULTADOS:")
    for item in resultados:
        print("-", item)

    # 🧩 Etapa 2 — Reenvia resultados ao modelo para conclusão
    contexto = "\n".join(resultados)
    prompt_final = f"""
    Aqui estão os dados obtidos das ferramentas:

    {contexto}

    Agora responda à pergunta original de forma completa e contextualizada:
    {query}
    """

    resposta_final = agente.llm.invoke(prompt_final)
    print("\n💬 Resposta final do modelo:")
    print(resposta_final.content)
