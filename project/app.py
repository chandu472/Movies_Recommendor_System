import pandas as pd
import numpy as np

from pickle import load
import streamlit as st
from PIL import Image

st.set_page_config(page_title='Movie Recommendation System', page_icon='📽️',layout='wide')
st.title(':red[Movie Recommendation System]')

image = Image.open('model/movies.jpg')
st.image(image)

a = st.selectbox('Select the type of Recommendation System', ('Content Based', 'Collaborative Based', 'Popularity Based'))
if a == 'Content Based':
    st.header(':red[Content Based Recommendation System]')

    sig = load(open('model/sig.pkl','rb'))
    df_act = load(open('model/df.pkl','rb'))

    # Reverse mapping of indices and movie titles
    indices = pd.Series(df_act.index, index = df_act['title']).drop_duplicates()

    movie_name = st.selectbox('Type or Select Movie from the Dropdown',
                                  df_act['title'])

    if st.button('Show Recommendation'):
        # Get the index corresponding to original_title
        idx = indices[movie_name]

        # Get the pairwsie similarity scores
        sig_scores = list(enumerate(sig[idx]))

        # Sort the movies
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

        # Scores of the 10 most similar movies
        sig_scores = sig_scores[1:11]

        # Movie indices
        movie_indices = [i[0] for i in sig_scores]

        # Top 10 most similar movies
        for i in df_act['title'].iloc[movie_indices]:
            st.text(i)

if a == 'Collaborative Based':
    st.header(':red[Collaborative Filtering Recommendation System]')

    model_knn = load(open('model/knn_model.pkl', 'rb'))

    df = load(open('model/collaborative_cosine_similarity.pkl', 'rb'))
    df = pd.DataFrame(df)

    st.subheader(':orange[Top 10 Movies]')

    movie_name = st.selectbox('Type or Select Movie from the Dropdown',
                              df.index.values)

    if st.button('Show Recommendation'):
        distances, indices = model_knn.kneighbors(df.loc[movie_name,:].values.reshape(1, -1), n_neighbors = 11)
        for i in range(0, len(distances.flatten())):
            if i == 0:
                st.text('Recommendations for ' + movie_name)
            else:
                st.text('{0}: {1}'.format(i, df.index[indices.flatten()[i]]))

if a == 'Popularity Based':
    st.header(':red[Popularity Based Recommendation System]')
    weight_average = load(open('model/weight_average.pkl','rb'))
    movies_popularity = load(open('model/movies_popularity.pkl','rb'))

    weight_average = weight_average[['title']]
    movies_popularity = movies_popularity[['title']]

    filter = st.selectbox('Select which type of Recommendation', ('Popularity','Average Votes'))

    if st.button('Recommend'):
        if filter == 'Popularity':
            st.header(':orange[Top 10 Popular Movies]')
            for i in movies_popularity['title'][0:10]:
                st.text(i)
        if filter == 'Average Votes':
            st.header(':orange[Best Movies by Average Votes]')
            for i in weight_average['title'][0:10]:
                st.text(i)