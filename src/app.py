import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import py7zr
import joblib
import os

film_data = pd.DataFrame()
recommended_films = []
similarity = None
genres = []
crews = []
setup = False
nFilmsRecomended = 0
unique_genres = set()
unique_genres_list = []

def my_setup():
    global setup
    global genres
    global crews 
    global nFilmsRecomended
    global unique_genres_list
    global similarity    
    global film_data
    
    if not os.path.exists('../models/similarity.pkl'): decompress()
    film_data = pd.read_csv('../data/processed/total_data_clean_procesed.csv')
    similarity = joblib.load('../models/similarity.pkl')
    
    genres = film_data.genres.values
    crews = film_data.crew.values
    nFilmsRecomended = 1
    genres = [genre for genre in genres if isinstance(genre, str)]
    
    for item in genres:
        unique_genres.update(item.split())
    unique_genres_list = list(unique_genres)
    unique_genres_list.sort()    
        
    setup = True

def decompress():
    #Descompresion

    z_file_path = '../models/models.7z'
    print("filename_path: "+z_file_path)
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
    #Fin descompresion


def recommend(movie):
        recomended_list = []
        try:    
        
            movie_index = film_data[film_data["title"] == movie].index[0]
            distances = similarity[movie_index]
            movie_list = sorted(list(enumerate(distances)), reverse = True , key = lambda x: x[1])[1:nFilmsRecomended+1]
        
            for i in movie_list:
                print(film_data.iloc[i[0]].title)
                recomended_list.append(film_data.iloc[i[0]].title)
            return recomended_list
        except Exception as e:
            print("Your movie is not in the List")
            
def contiene_todos_los_generos(row, selected_genres):
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

def main():
    # Filtrar la lista de géneros para eliminar elementos que no son cadenas de texto
    global recommended_films
    global nFilmsRecomended

    
    if not setup: my_setup()
       
    st.header('Que peli vas a ver hoy?!')

    nFilmsRecomended = st.slider(min_value=1,max_value=5,label='Cuantas pelis quieres que te recomiende?')
    selected_genre = st.multiselect("Selecciona un género:", unique_genres_list)
    st.text('Esto filtrara las peliculas disponibles en la lista inferior')

    # Aplicar la función a cada fila del DataFrame
    film_data['cumple_condicion'] = film_data.apply(lambda row: contiene_todos_los_generos(row, selected_genre), axis=1)
    peliculas_filtradas = film_data[film_data['cumple_condicion']]

    selected_film = st.selectbox("Indica una pelicula que te guste como orientación:", peliculas_filtradas.title.values)

    if st.button("RECOMENDAR"):
       recommended_films = recommend(selected_film)
       
    #if recommended_films:
    for string in recommended_films:
        st.markdown(f"### {string}")
        
        
if __name__ == '__main__':
    main()