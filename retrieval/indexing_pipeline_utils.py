from __future__ import annotations

from langchain_core.documents import Document

from data_utils import Movie
from retrieval import config


def create_docs_to_embedd(movies: list[Movie], config: config.RetrievalExpsConfig) -> list[Document]:
    """
    Convierte una lista de objetos `Movie` a una lista the objetos `Document`(usada por Langchain).
    En esta función se decide que parte de los datos será usado como embeddings y que parte como metadata.
    """
    movies_as_docs = []
    for movie in movies:
        content = config.text_to_embed_fn(movie)
        metadata = movie.model_dump()
        doc = Document(page_content=content, metadata=metadata)
        movies_as_docs.append(doc)

    return movies_as_docs


## Posibles funciones para usar como `text_to_embed_fn` en `RetrievalExpsConfig` ##


def get_synopsys_txt(movie: Movie) -> str:
    return movie.synopsis

def get_full_info(movie: Movie) -> str:
    # Incluye título, año, géneros, actores, y resumen
    genres = ", ".join(movie.genre_tags) if movie.genre_tags else ""
    actors = ", ".join(movie.cast_top_5) if movie.cast_top_5 else ""
    return f"Película: {movie.title_es} ({movie.year})\nGéneros: {genres}\nActores: {actors}\nResumen: {movie.synopsis}"

def get_detailed_movie_info(movie: Movie) -> str:
    actors = ", ".join(movie.cast_top_5) if hasattr(movie, 'actors') and movie.cast_top_5 else ""
    genres = ", ".join(movie.genre_tags) if movie.genre_tags else ""
    director = movie.director_top_5 if hasattr(movie, 'director') else ""
    year = movie.year if hasattr(movie, 'year') else ""
    country = movie.country if hasattr(movie, 'country') else ""
    return (
        f"Título: {movie.title_es} ({year})\n"
        f"Director: {director}\n"
        f"País: {country}\n"
        f"Géneros: {genres}\n"
        f"Actores: {actors}\n"
        f"Sinopsis: {movie.synopsis}"
    )

def get_director_and_country(movie: Movie) -> str:
    director = movie.director_top_5 if hasattr(movie, 'director') else ""
    country = movie.country if hasattr(movie, 'country') else ""
    return f"Director: {director}. País: {country}. Sinopsis: {movie.synopsis}"