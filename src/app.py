#https://fourgeeks-streamlit-integration-milser.onrender.com/

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import py7zr
import joblib
import os


class FilmRecommender:
    def __init__(self):
        self.film_data = pd.DataFrame()
        self.recommended_films = []
        self.similarity = None
        self.genres = []
        self.crews = []
        self.setup = False
        self.n_films_recomended = 0
        self.unique_genres = set()
        self.unique_genres_list = []

    def my_setup(self):
        if not os.path.exists('../models/similarity.pkl'):
            self.decompress()
        self.film_data = pd.read_csv('../data/processed/total_data_clean_procesed.csv')
        self.similarity = joblib.load('../models/similarity.pkl')
        
        self.genres = self.film_data.genres.values
        self.crews = self.film_data.crew.values
        self.n_films_recomended = 1
        self.genres = [genre for genre in self.genres if isinstance(genre, str)]
        
        for item in self.genres:
            self.unique_genres.update(item.split())
        self.unique_genres_list = list(self.unique_genres)
        self.unique_genres_list.sort()    
        
        self.setup = True

    def decompress(self):
        # Descompresion
        z_file_path = '../models/models.7z'
        print("filename_path: " + z_file_path)
        # Ruta del archivo .7z
        archive_path = z_file_path
        # Ruta del directorio de destino
        output_dir = '../models'

        # Extracción del archivo
        try:
            with py7zr.SevenZipFile(archive_path) as z:
                z.extractall(path=output_dir)
            print("Extracción completada con éxito.")
        except Exception as e:
            print("Ocurrió un error durante la extracción:")
            print(e)
        # Fin descompresion

    def recommend(self, movie):
        recomended_list = []
        try:
            movie_index = self.film_data[self.film_data["title"] == movie].index[0]
            distances = self.similarity[movie_index]
            movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:self.n_films_recomended + 1]

            for i in movie_list:
                print(self.film_data.iloc[i[0]].title)
                recomended_list.append(self.film_data.iloc[i[0]].title)
            return recomended_list
        except Exception:
            print("Your movie is not in the List")
            return []

    def contiene_todos_los_generos(self, row, selected_genres):
        # Dividir la cadena de géneros en una lista de géneros
        if isinstance(row['genres'], str):
            lista_de_generos = row['genres'].split()
            # Verificar si todos los géneros seleccionados están presentes en la lista de géneros
            for genre in selected_genres:
                if genre not in lista_de_generos:
                    return False
            return True
        else:
            return False
        
@st.cache_resource
def get_film_recommender():
    film_recommender = FilmRecommender()
    film_recommender.my_setup()
    return film_recommender

def main():
    film_recommender = get_film_recommender()

    print(film_recommender.unique_genres_list)

    st.header('Que peli vas a ver hoy?!')

    film_recommender.n_films_recomended = st.slider(min_value=1, max_value=5, label='Cuantas pelis quieres que te recomiende?')
    selected_genre = st.multiselect("Selecciona un género:", film_recommender.unique_genres_list)
    st.text('Esto filtrara las peliculas disponibles en la lista inferior')

    # Aplicar la función a cada fila del DataFrame
    film_recommender.film_data['cumple_condicion'] = film_recommender.film_data.apply(
        lambda row: film_recommender.contiene_todos_los_generos(row, selected_genre), axis=1)
    peliculas_filtradas = film_recommender.film_data[film_recommender.film_data['cumple_condicion']]

    selected_film = st.selectbox("Indica una pelicula que te guste como orientación:", peliculas_filtradas.title.values)

    recommended_films = []
    if st.button("RECOMENDAR"):
        recommended_films = film_recommender.recommend(selected_film)

    for string in recommended_films:
        st.markdown(f"### {string}")

if __name__ == '__main__':
    main()
