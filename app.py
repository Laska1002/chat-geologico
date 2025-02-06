import streamlit as st
from langchain_ollama.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import AgentType

# Configuración de la base de datos SQLite
url_query = "sqlite:///geodata.db"  # Ruta de la base de datos SQLite
db = SQLDatabase.from_uri(url_query, sample_rows_in_table_info=3)

# Inicializar el modelo LLM con Ollama
llm = ChatOllama(model="llama3.1:latest", temperature=0.5)

# Crear el agente SQL con LangChain
chain = create_sql_agent(
    db=db,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    agent_executor_kwargs={"handle_parsing_errors": True},
)

# Función para limpiar la consulta SQL
def clear_query(query):
    prompt = (
        "Given the following text containing a SQL query.\n"
        f"{query}\n"
        "Your job is to extract only the SQL query from the text. Without modifying it.\n"
        "No chit chat is needed.\n"
        "Just answer with the SQL Query."
    )
    response = llm.invoke(prompt)
    return response.content

# Función para generar la respuesta a la pregunta del usuario
def generate_answer(query, question, result):
    prompt = (
        "Given the following user question about Geology Thesis papers, corresponding SQL query, "
        "and SQL result, answer the user question always mentioning the Geology Thesis papers as main topic.\n\n"
        "The answer needs to be in Spanish.\n\n"
        f"Question: {question}\n"
        f"SQL Query: {query}\n"
        f"SQL Result: {result}"
    )
    response = llm.invoke(prompt)
    return response.content

# Configuración de la interfaz en Streamlit
st.title("🔍 Chat Geológico con Ollama")
st.write("Escribe tu consulta sobre geología y obtén resultados desde la base de datos.")

# Entrada del usuario
user_question = st.text_input("Escribe tu pregunta:", "")

if st.button("Enviar"):
    if user_question:
        try:
            # Generar la consulta SQL y obtener los resultados
            query = chain.run(input=user_question)
            query = clear_query(query)
            result = db.run(query)

            # Generar la respuesta final
            answer = generate_answer(query=query, question=user_question, result=result)

            # Mostrar los resultados
            st.subheader("Respuesta:")
            st.write(answer)
        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")
    else:
        st.warning("⚠️ Por favor, escribe una pregunta.")
