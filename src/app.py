import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
import py7zr
import joblib
import psutil
import os
import pathlib

dirname = str(pathlib.Path(__file__).parent.parent)


print("dirname:" + dirname)


st.header('Que peli vas a ver hoy?!')

nFilmsRecomended = st.slider(min_value=1,max_value=5,label='Cuantas pelis quieres que te recomiende?')

#film_data = pd.read_csv(r'../data/processed/total_data_clean_procesed.csv')
film_data = pd.read_csv(dirname + r'\data\processed\total_data_clean_procesed.csv')


#Descompresion

z_file_path = dirname + '\models\models.7z'
print("filename_path: "+z_file_path)
# Ruta del archivo .7z
archive_path = z_file_path
# Ruta del directorio de destino
output_dir = dirname + r'\models'
# Número de núcleos menos cuatro
num_cores = psutil.cpu_count(logical=True) - 1 if psutil.cpu_count(logical=True) > 1 else 1
# Asegurarse de que el número de núcleos no sea menos de 1
num_cores = max(1, num_cores)

# Configurar py7zr para usar varios hilos (actualmente py7zr no soporta multithreading directo,
# pero lo incluimos para ilustrar cómo se podría adaptar en el futuro o si la biblioteca añade soporte)
# Por ahora, py7zr no tiene una opción nativa para especificar el número de núcleos, se puede considerar en futuras versiones.

# Extracción del archivo
try:
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=output_dir)
    print("Extracción completada con éxito.")
except Exception as e:
    print("Ocurrió un error durante la extracción:")
    print(e)


#Fin descompresion


# Cargar el vectorizador
vectorizer = joblib.load(r'C:\Users\milser\Documents\Trasteo_4geeks\Pagina-Web-de-ML-con-Streamlit\models\vectorizer.pkl')
# Cargar la matriz de similitud
similarity = joblib.load(r'C:\Users\milser\Documents\Trasteo_4geeks\Pagina-Web-de-ML-con-Streamlit\models\similarity.pkl')

genres = film_data.genres.values
crews = film_data.crew.values

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
    
genres = [genre for genre in genres if isinstance(genre, str)]
unique_genres = set()

def main():
    # Filtrar la lista de géneros para eliminar elementos que no son cadenas de texto
    for item in genres:
        unique_genres.update(item.split())
    unique_genres_list = list(unique_genres)
    unique_genres_list.sort()


    selected_genre = st.multiselect("Selecciona un género:", unique_genres_list)
    st.text('Esto filtrara las peliculas disponibles en la lista inferior')

   

    # Aplicar la función a cada fila del DataFrame
    film_data['cumple_condicion'] = film_data.apply(lambda row: contiene_todos_los_generos(row, selected_genre), axis=1)
    peliculas_filtradas = film_data[film_data['cumple_condicion']]
    peliculas_filtradas = peliculas_filtradas.drop(columns=['cumple_condicion'])

    selected_film = st.selectbox("Indica una pelicula que te guste como orientación:", peliculas_filtradas.title.values)

    for string in recommend(selected_film):
        st.markdown(f"### {string}")
        
        
if __name__ == '__main__':
    main()