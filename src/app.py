import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import py7zr
import joblib
import multiprocessing

st.header('Que peli vas a ver hoy?!')

nFilmsRecomended = st.slider(min_value=1,max_value=5,label='Cuantas pelis quieres que te recomiende?')

#film_data = pd.read_csv(r'../data/processed/total_data_clean_procesed.csv')
film_data = pd.read_csv(r'C:\Users\milser\Documents\Trasteo_4geeks\Pagina-Web-de-ML-con-Streamlit\data\processed\total_data_clean_procesed.csv')

# Cargar el vectorizador
vectorizer = joblib.load(r'C:\Users\milser\Documents\Trasteo_4geeks\Pagina-Web-de-ML-con-Streamlit\models\vectorizer.pkl')
# Cargar la matriz de similitud
similarity = joblib.load(r'C:\Users\milser\Documents\Trasteo_4geeks\Pagina-Web-de-ML-con-Streamlit\models\similarity.pkl')

genres = film_data.genres.values
crews = film_data.crew.values

# Filtrar la lista de géneros para eliminar elementos que no son cadenas de texto
genres = [genre for genre in genres if isinstance(genre, str)]
unique_genres = set()
for item in genres:
    unique_genres.update(item.split())
unique_genres_list = list(unique_genres)
unique_genres_list.sort()

selected_genre = st.multiselect("Selecciona un género:", unique_genres_list)
st.text_area = "ads"

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

# Aplicar la función a cada fila del DataFrame
film_data['cumple_condicion'] = film_data.apply(lambda row: contiene_todos_los_generos(row, selected_genre), axis=1)
peliculas_filtradas = film_data[film_data['cumple_condicion']]
peliculas_filtradas = peliculas_filtradas.drop(columns=['cumple_condicion'])

selected_film = st.selectbox("Indica una pelicula que te guste como orientación:", peliculas_filtradas.title.values)


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


for string in recommend(selected_film):
    st.markdown(f"### {string}")