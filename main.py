import pandas as pd
import gc
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity

from fastapi import FastAPI

# Se instancia la app
app = FastAPI()

# Se cargan los dataset
df_games = pd.read_parquet('steam_games.parquet')
df_reviews = pd.read_parquet('user_reviews.parquet')
df_items = pd.read_parquet('user_items_mitad.parquet')
df_modelo = pd.read_parquet('dataset_ml.parquet')


#@app.get('/')
#def root():
#    """ Mensaje de bienvenida """
    
#    return {"message" : "Bienvenidos!"}


@app.get('/PlayTimeGenre/{genero}')
def PlayTimeGenre( genero : str ):
    """  Debe devolver el año con más horas jugadas para dicho género.

Ejemplo de retorno: 

{"Año de lanzamiento con más horas jugadas para Género X" : 2013}"""

    if genero not in df_games.columns:
        # Se imprime mensaje de error:   
        return f"ERROR: El género {genero} no existe en la base de datos."     
    else:
         # Extraigo del df_games todos aquellos juegos catalogados dentro del género dado:
        df_filtrado = df_games[df_games[genero] == 1]

        # Se seleccionan las columnas necesarias de los dataframes df_filtrado y df_items:
        df_playtimegenre = pd.merge(df_filtrado[['Game_id','Release_year']], df_items[['Game_id',"Playtime"]], 
                            on="Game_id", how = 'inner')
        
        # Reviso que el resultado del merge no sea un dataframe vacío
        if df_playtimegenre.shape[0] == 0:
            return f"No hay información de horas de juego para el género {genero}." 
        else:

            # Se agrupa el df por Release_year sumando la cantidad de horas jugadas y buscando el año con el valor máximo:
            horas_por_anio = df_playtimegenre.groupby("Release_year")["Playtime"].sum()
            anios_ordenados = horas_por_anio.sort_values(ascending=False).head(1).index.to_list()
            anio_max=anios_ordenados[0]
    
            # Se elimina la basura para liberar uso de memoria:
            gc.collect()

            # Se crea la clave que voy a usar en el diccionario de resultado:
            clave = f'Año de lanzamiento con más horas jugadas para Género {genero}: '

    return {clave:anio_max}
    

@app.get('/UserForGenre/{genero}')

def UserForGenre( genero : str ):
    """ Debe devolver el usuario que acumula más horas jugadas para el género dado y una 
    lista de la acumulación de horas jugadas por año.
   
   Ejemplo de retorno: 
   
   {"Usuario con más horas jugadas para Género X" : us213ndjss09sdf, "Horas jugadas":[{Año: 2013, Horas: 203}, {Año: 2012, Horas: 100}, {Año: 2011, Horas: 23}]} """

    # Se imprime mensaje de error: 
    if genero not in df_games.columns:   
        return f"ERROR: El género {genero} no existe en la base de datos."  
    
    else:
        # Extraigo del df_games todos aquellos juegos catalogados dentro del género dado:
        df_filtrado = df_games[df_games[genero] == 1]

        # Se seleccionan las columnas necesarias de los dataframes df_filtrado y df_items:
        df_usergenre = pd.merge(df_filtrado[['Game_id','Release_year']], df_items[['Game_id',"Playtime", 'User_id']], 
                            on="Game_id", how = 'inner')
        
        # Reviso que el resultado del merge no sea un dataframe vacío
        if df_usergenre.shape[0] == 0:
            return f"No hay información de usuarios con horas de juego para el género {genero}." 
        else:   
            # Se agrupa el df por User_id sumando la cantidad de horas jugadas y buscando el usuario con el valor máximo:
            usuario_max = df_usergenre.groupby("User_id")["Playtime"].sum().idxmax()

            #Extraigo la información correspondiente a ese usuario que tiene el valor máximo de horas jugadas:
            df_usergenre = df_usergenre[df_usergenre["User_id"] == usuario_max] 

            # Se agrupa la cantidad de horas jugadas por año por el usuario:
            horas_anio = df_usergenre.groupby("Release_year")["Playtime"].sum()

            # Se convierte este resultado a un diccionario:
            horas_dicc = horas_anio.to_dict() 

            # Se crea un diccionario vacío que almacenará los valores con el formato con que los voy a entregar:
            horas_dicc1 = {}
                
            # Se itera sobre cada uno de los pares clave-valor del diccionario original:
            for clave, valor in horas_dicc.items(): 
                formato_clave = f'Año: {int(clave)}'           # se da formato al año
                formato_valor = f'Horas: {int(valor)}'       # se da formato a la cantidad de horas jugadas
                horas_dicc1[formato_clave] = formato_valor      # se asignan los valores al nuevo diccionario creado

            # Se crea la clave que voy a usar en el diccionario de resultado:
            clave_dicc = f'Usuario con más horas jugadas para Género {genero}'
        
            # Se elimina la basura para liberar uso de memoria:
            gc.collect()
        
            # Se retornan los valores en un diccionario: 
            return {clave_dicc : usuario_max, "Horas jugadas": horas_dicc1}


@app.get('/UsersRecommend/{anio}')
def UsersRecommend( anio : int ):

    """     Devuelve el top 3 de juegos MÁS recomendados por usuarios para el año dado. 
            (reviews.recommend = True y comentarios positivos/neutrales)

            Ejemplo de retorno: 
            
            [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}] """

    yearmin=df_reviews['Posted_year'].min()
    yearmax=df_reviews['Posted_year'].max()
    if anio < yearmin or anio > yearmax: 
        # Se imprime mensaje de error: 
        return f"ERROR: No hay recomendaciones de usuarios para el año {anio}"  
    
    else:
        # Extraigo del df_reviews todos aquellos registros que corresponden al año de publicacion dado:
        df_filtrado = df_reviews[df_reviews['Posted_year'] == anio]

        # Creo una nueva columna que combina Recommend y Sentiment analysis multiplicándolas, con eso si
        # el valor de alguna de las dos es 0 no será tenida en cuenta luego en la suma de las recomendaciones
        df_filtrado['Combinada'] = df_filtrado['Recommend']*df_filtrado['Sentiment_analysis']

        # Hago merge con df_games seleccionando las columnas que necesito:
        df_usersrecommend = pd.merge(df_filtrado[['Game_id', 'Posted_year','Combinada']], 
                                     df_games[['Game_id', 'Name']], on = "Game_id", how = 'inner')

        # Reviso que el resultado del merge no sea un dataframe vacío
        if df_usersrecommend.shape[0] == 0:
            return f"ERROR: No hay recomendaciones ni reseñas positivas de usuarios para el año {anio}" 
        else: 

            # Agrupo por nombre del juego y sumo los valores de la columna combinada:
            df_usersrecommend = df_usersrecommend.groupby('Name')['Combinada'].sum()

            # Se ordenan las recomendaciones por orden descendente, se seleccionan las primeras 3:
            mas_recomendados = df_usersrecommend.sort_values(ascending=False).head(3).index.to_list()

            # Elaboro el diccionario de salida
            dicc = {}
            if len(mas_recomendados)>=3:
                dicc['Puesto 1'] = mas_recomendados[0]
                dicc['Puesto 2'] = mas_recomendados[1]
                dicc['Puesto 3'] = mas_recomendados[2]
            elif len(mas_recomendados)==2:
                dicc['Puesto 1'] = mas_recomendados[0]
                dicc['Puesto 2'] = mas_recomendados[1]
                dicc['Puesto 3'] = 'Sin datos'
            elif len(mas_recomendados)==1:
                dicc['Puesto 1'] = mas_recomendados[0]
                dicc['Puesto 2'] = 'Sin datos'
                dicc['Puesto 3'] = 'Sin datos'

        # Se elimina la basura para liberar uso de memoria:
        gc.collect()

    return dicc

@app.get('/UsersWorstDeveloper/{anio}')
def UsersWorstDeveloper( anio : int ):
    """ Devuelve el top 3 de desarrolladoras con juegos MENOS recomendados por usuarios para el año dado. 
    (reviews.recommend = False y comentarios negativos)
    
    Ejemplo de retorno: 
    
    [{"Puesto 1" : X}, {"Puesto 2" : Y},{"Puesto 3" : Z}] """

    yearmin=df_reviews['Posted_year'].min()
    yearmax=df_reviews['Posted_year'].max()
    if anio < yearmin or anio > yearmax:  
        # Se imprime mensaje de error: 
        return f"ERROR: No hay recomendaciones de usuarios para el año {anio}"  
    
    else:
        # Extraigo del df_reviews todos aquellos registros que corresponden al año de publicacion dado:
        df_filtrado = df_reviews[df_reviews['Posted_year'] == anio]

        # Creo una nueva columna que combina Recommend y Sentiment analysis multiplicándolas
        df_filtrado['Combinada'] = df_filtrado['Recommend']*df_filtrado['Sentiment_analysis']

        # Selecciono únicamente los registros con malos reviews, es decir los que tienen 0 en Combinada:
        df_negativos = df_filtrado[df_filtrado['Combinada'] == 0]

        # Hago merge con df_games seleccionando las columnas que necesito:
        df_usersrecommend = pd.merge(df_negativos[['Game_id', 'Posted_year','Combinada']], 
                                     df_games[['Game_id', 'Developer']], on = "Game_id", how = 'inner')

        # Agrupo por id y sumo los valores de la columna combinada
        df_usersrecommend = df_usersrecommend.groupby('Developer')['Combinada'].count()

        # Se ordena el conteo de malas reseñas por orden descendente, se seleccionan las primeras 3:
        menos_recomendados = df_usersrecommend.sort_values(ascending=False).head(3).index.to_list()

        # Elaboro el diccionario de salida:
        dicc = {}
        if len(menos_recomendados)>=3:
            dicc['Puesto 1'] = menos_recomendados[0]
            dicc['Puesto 2'] = menos_recomendados[1]
            dicc['Puesto 3'] = menos_recomendados[2]
        elif len(menos_recomendados)==2:
            dicc['Puesto 1'] = menos_recomendados[0]
            dicc['Puesto 2'] = menos_recomendados[1]
            dicc['Puesto 3'] = 'Sin datos'
        elif len(menos_recomendados)==1:
            dicc['Puesto 1'] = menos_recomendados[0]
            dicc['Puesto 2'] = 'Sin datos'
            dicc['Puesto 3'] = 'Sin datos'
        else:
            return f"ERROR: No hay recomendaciones negativas de usuarios para el año {anio}"

        # Se elimina la basura para liberar uso de memoria:
        gc.collect()

    return dicc

@app.get('/sentiment_analysis/{desarrollador}')
def sentiment_analysis( desarrollador: str ):
    """     Según la empresa desarrolladora, se devuelve un diccionario con el nombre de la desarrolladora 
    como llave y una lista con la cantidad total de registros de reseñas de usuarios que se encuentren 
    categorizados con un análisis de sentimiento como valor.

Ejemplo de retorno: 

{'Valve' : [Negative = 182, Neutral = 120, Positive = 278]} """

    if desarrollador not in df_games['Developer'].values:
        # Mensaje de error si no encuentra el desarrollador dado:
        return f"ERROR: El desarrollador {desarrollador} no existe en la base de datos."
    else:
        #Extraigo del df_games los registros correspondientes al desarrollador:
        df_filtrado = df_games[df_games['Developer']==desarrollador]

        # Hago merge con df_reviews seleccionando las columnas que necesito:
        df_sentiment = pd.merge(df_filtrado[['Game_id', 'Developer']], df_reviews[['Game_id', 'Sentiment_analysis']],
                                                on = "Game_id", how = 'inner')
        
        # Extraigo los reviews positivos, es decir con valor 2 en Sentiment analysis y los cuento
        positivos = df_sentiment[df_sentiment['Sentiment_analysis']==2].shape[0]

        # Extraigo los reviews neutros, es decir con valor 1 en Sentiment analysis y los cuento
        neutros = df_sentiment[df_sentiment['Sentiment_analysis']==1].shape[0]

        # Extraigo los reviews negativos, es decir con valor 0 en Sentiment analysis y los cuento
        negativos = df_sentiment[df_sentiment['Sentiment_analysis']==0].shape[0]

        # Armo el string de salida:
        resultados = f"[Negative = {negativos}, Neutral = {neutros}, Positive = {positivos}]"

        return {desarrollador:resultados}
    
@app.get('/recomendacion_juego/{id}')
def recomendacion_juego( id : float): 
    
    """ Ingresando el id de producto, deberíamos recibir una lista con 5 juegos 
    recomendados similares al ingresado. """

    #tomo un dataframe auxiliar sólo con las columnas reviews y item_id
    df = df_modelo.drop(columns = 'item_name')
    
    #agrupo los reviews por juegos
    df_modelo["review"] = df_modelo["review"].fillna("")
    grouped = df.groupby('item_id').agg(lambda x: ' '.join(x)).reset_index()
    
    #Vectorización de términos mediante tf-idf, usando stop words en inglés
    tfidf_vectorizer = TfidfVectorizer(stop_words = 'english')

    #Aplica el vectorizador tf-idf a la columna 'review' del dataframe agrupado
    tfidf_matrix = tfidf_vectorizer.fit_transform(grouped['review'])

    #Calcula la similitud del coseno entre las reseñas utilizando linear_kernel
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    #busca el indice del item de entrada de la función
    idx = grouped.index[grouped['item_id']== id].tolist()[0]

    #Se crea una lista de tuples, donde cada tuple contiene el índice del artículo 
    #y su puntaje de similitud de coseno con el artículo específico.
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Se ordena descendente, de manera que las reseñas más similares aparezcan primero.
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse = True)

    #selecciono las 5 primeras
    sim_scores = sim_scores[1:6]
    item_indices = [i[0] for i in sim_scores]
    
    #obtiene la lista de item_id correspondiente a los índices
    salida_ids = grouped['item_id'].iloc[item_indices].tolist()

    #obtiene la lista de nombres
    salida_nombres = df_modelo.loc[df['item_id'].isin(salida_ids),'item_name'].unique().tolist()
    
    return salida_nombres