from langchain_core.tools import tool

from typing import List
from datetime import datetime, timedelta
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv

from app.storage.vector_storage import VectorStorage

load_dotenv()


vdb_config = {
    "index_name": "youtube-transcription-summary-openia-index",
    "embedding_model": OpenAIEmbeddings(model="text-embedding-3-small"),
}


vector_storage = VectorStorage(vdb_config)


def parse_fecha(fecha_usuario: str) -> str:
    """
    Convierte expresiones como 'hoy', 'ayer', o fechas explícitas en formato 'YYYY-MM-DD'.
    """
    fecha_usuario = fecha_usuario.strip().lower()
    hoy = datetime.now().date()
    if fecha_usuario in ["hoy", "today"]:
        return hoy.strftime("%Y-%m-%d")
    elif fecha_usuario in ["ayer", "yesterday"]:
        return (hoy - timedelta(days=1)).strftime("%Y-%m-%d")
    else:
        # Intenta parsear fecha explícita
        try:
            # Admite formatos comunes, ajusta según tus inputs
            return (
                datetime.strptime(fecha_usuario, "%Y-%m-%d").date().strftime("%Y-%m-%d")
            )
        except Exception:
            raise ValueError(f"Formato de fecha no reconocido: {fecha_usuario}")


# @tool("buscar_documentos_fecha",
#       return_direct=False,
#       args_schema=None,
#       description=(
#           "Úsala para buscar documentos en la base Pinecone relacionados con una consulta de texto y filtrados por una fecha. "
#           "El argumento 'fecha' admite las palabras 'hoy', 'ayer', o una fecha precisa con formato 'YYYY-MM-DD'."
#       )
# )
@tool
def buscar_documentos_fecha(
    query: str,
    fecha: str = None,
) -> List[str]:
    """
    Busca documentos relevantes con Pinecone y filtra solo por fecha.

    Args:
        query (str): El texto de la consulta para buscar documentación relevante.
        fecha (str): Fecha a filtrar, puede ser 'hoy', 'ayer', o una fecha específica en formato 'YYYY-MM-DD'.
    Returns:
        List[str]: Lista de textos/documentos relevantes encontrados.
    """
    if fecha:
        try:
            fecha_valida = parse_fecha(fecha)
        except Exception as e:
            fecha_valida = datetime.now().strftime("%Y-%m-%d")
        print(fecha_valida)

        filtro_metadata = {"publish_date": fecha_valida}
    else:
        filtro_metadata = None
    docs = vector_storage.search(query=query, k=5, metadata_filter=filtro_metadata)
    return [doc.page_content for doc in docs]
