# Importation des bibliothèques nécessaires
import re
import dill
import nltk
import joblib
import html5lib
import numpy as np
import streamlit as st
from nltk import pos_tag
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Téléchargement des ressources nécessaires de NLTK
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


# Définition du chemin d'accès aux ressources
path = 'ressources/'


# Chargement des fichiers
vectorizer_CV = joblib.load(path + 'countvectorizer.joblib')
mlb = joblib.load(path + 'multilabelbinarizer.joblib')

with open(path + 'stop_words.pkl', 'rb') as f:
    stop_words = dill.load(f)

with open(path + 'top_500_tags.pkl', 'rb') as f:
    top_500_tags = dill.load(f)

with open(path + 'pipelines.pkl', 'rb') as file:
    pipelines = dill.load(file)

with open(path + 'lda.pkl', 'rb') as file:
    lda = dill.load(file)


# Définir un dictionnaire de fonctions de modèles et de leurs paramètres associés
model_functions = {
    "LogisticRegression": {"function": pipelines["LogisticRegression"].predict},
    "SGDClassifier": {"function": pipelines["SGDClassifier"].predict},
    "CountVectorizer": {"function": pipelines["CountVectorizer"].transform}
}

# Définition de l'interface utilisateur
st.markdown(
    "<h1 style='margin-top: 0; padding-top: 0;'>Générateur de tags Stackoverflow</h1>",
    unsafe_allow_html=True)
subtitle = '<p style="font-size: 30px;">Catégorisez automatiquement des questions</p>'
st.markdown(subtitle, unsafe_allow_html=True)
button_style = "background-color: black; color: white; border-radius: 5px;"


# Sélection du modèle à utiliser
st.sidebar.header("Models")
model_choice = None
with st.sidebar.container():
    model_choice = st.selectbox(" ", model_functions.keys())

# Saisie du titre et du texte à utiliser
title = st.text_input("Your Title goes here :")
post = st.text_area("Your Text goes there :", height=250)


# Si aucune approche ni aucun modèle ne sont sélectionnés, afficher un message d'erreur
if model_choice is None:
    st.error("Please pick a model")

else:

    # Génération des tags si l'utilisateur a cliqué sur le bouton et a fourni des données
    if st.button("Tags") and title and post and (model_choice is not None or ""):

        # Concaténer le titre et le message en une seule chaîne
        user_input = title + " " + post

        # Récupérer la fonction pour les modèles
        model_function = model_functions[model_choice]["function"]
        if model_choice == "CountVectorizer":
            tag_transform = lambda output: list(t[0] for t in output[0])
        else:
            tag_transform = lambda output: list(mlb.inverse_transform(output)[0])        

        # Appliquer le modèle choisi à la chaîne d'entrée
        output = model_function(user_input)

        # Extraire les tags prédits de la sortie
        tags = tag_transform(output)

        # Impression des tags
        buttons = "  ".join([f'<button style="{button_style}">{text}</button>' for text in tags])
        st.markdown(buttons, unsafe_allow_html=True)
