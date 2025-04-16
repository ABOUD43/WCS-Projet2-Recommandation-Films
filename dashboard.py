import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

# Configuration de la page
st.set_page_config(
    page_title="Recommandation de Film",
    layout="wide",
    page_icon=":clapper:"
)

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def get_similar_titles(df, title):
 
    base_title = title.lower().split(' and ')[0].strip() 
    return df[df['primaryTitle'].str.lower().str.contains(base_title, na=False)]



@st.cache_data
def prepare_features(df_encoded):
    X = df_encoded.select_dtypes(include=['int64', 'float64']).drop(columns=['startYear', 'Decade', 'revenue', 'budget'])
    numerical_cols = ['runtimeMinutes', 'averageRating', 'numVotes', 'popularity']
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    return X_scaled, scaler

@st.cache_resource
def train_model(X_scaled):
    modelNN = NearestNeighbors(n_neighbors=10,algorithm='auto',metric='euclidean',n_jobs=-1)
    modelNN.fit(X_scaled)
    return modelNN

def afficher_film(row, list_genres, list_languages):
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if pd.notna(row['poster_path']):
            img_url = f"https://image.tmdb.org/t/p/w500/{row['poster_path']}"
            st.image(img_url, width=200)
    
    with col2:
        st.markdown(f"### 🎬 {row['primaryTitle']} ({int(row['startYear'])})")
        if row.get('overview') and row['overview'] != 'unknown':
            st.markdown(f"**📝 Synopsis :** {row['overview']}")
        genres_present = [genre for genre in list_genres if row.get(f"genre_{genre}", False)]
        if genres_present:
            st.markdown(f"**🎭 Genres :** {', '.join(genres_present)}")
        languages_present = [lang for lang in list_languages if row.get(lang, False)]
        if languages_present:
            st.markdown(f"**🌐 Langue :** {', '.join(languages_present)}")
        if row['averageRating']:
            st.markdown(f"**⭐ Note :** {round(row['averageRating'], 2)}/10 ({int(row['numVotes'])} votes)")
        if pd.notna(row['contributors_directors_writers']):
            st.markdown(f"**🎥 Réalisation :** {row['contributors_directors_writers']}")
        if pd.notna(row['contributors_actors']):
            st.markdown(f"**🎭 Casting :** {row['contributors_actors']}")
        imdb_url = f"https://www.imdb.com/title/{row['tconst']}/"
        st.markdown(f"[🔗 Voir sur IMDB]({imdb_url})")
    st.markdown("---")

def create_sidebar():
    with st.sidebar:
        st.header("⚙️ Filtres avancés")
        
        # Filtres de notation
        st.subheader("📊 Notation")
        min_rating = st.slider("Note minimum", 0.0, 10.0, 5.0, 0.5)
        min_votes = st.number_input("Nombre minimum de votes", 0, 1000000, 1000)
        
        # Filtres de durée
        st.subheader("⏱️ Durée")
        duration_range = st.slider("Durée (minutes)",0, 300, (60, 180))
        
        return {
            'min_rating': min_rating,
            'min_votes': min_votes,
            'duration_range': duration_range
        }

def show_movie_stats(df):
    st.subheader("📈 Statistiques de la base de données")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Films disponibles", f"{len(df):,}")
    with col2:
        avg_rating = df['averageRating'].mean()
        st.metric("Note moyenne", f"{avg_rating:.2f}/10")
    with col3:
        recent_year = df['startYear'].max()
        st.metric("Film le plus récent", f"{int(recent_year)}")
    with col4:
        total_votes = df['numVotes'].sum()
        st.metric("Total des votes", f"{total_votes:,}")

# Définition des listes de genres et langues
list_genres = [
    'Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Musical', 'Mystery',
    'Romance', 'Sci-fi', 'Sport', 'Thriller', 'War', 'Western']

list_languages = [
    'Afrikaans', 'Akan', 'Albanian', 'Amharic', 'Arabic', 'Armenian', 'Assamese',
    'Aymara', 'Azerbaijani', 'Bambara', 'Bashkir', 'Basque', 'Belarusian', 'Bengali',
    'Bosnian', 'Bulgarian', 'Burmese', 'Catalan', 'Chechen', 'Chinese', 'Cree', 'Croatian',
    'Czech', 'Danish', 'Dhivehi', 'Dutch', 'Dzongkha', 'English', 'Esperanto', 'Estonian',
    'Faroese', 'Finnish', 'French', 'Fulah', 'Galician', 'Ganda', 'Georgian', 'German',
    'Guarani', 'Gujarati', 'Haitian', 'Hausa', 'Hebrew', 'Hindi', 'Hungarian', 'Icelandic',
    'Igbo', 'Indonesian', 'Interlingue', 'Inuktitut', 'Irish', 'Italian', 'Japanese',
    'Javanese', 'Kalaallisut', 'Kannada', 'Kashmiri', 'Kazakh', 'Khmer', 'Kinyarwanda',
    'Kirghiz', 'Kongo', 'Korean', 'Kurdish', 'Langue inconnue', 'Lao', 'Latin', 'Latvian',
    'Limburgan', 'Lithuanian', 'Luxembourgish', 'Macedonian', 'Malagasy', 'Malay (macrolanguage)',
    'Malayalam', 'Maltese', 'Maori', 'Marathi', 'Marshallese', 'Modern Greek (1453-)', 'Mongolian',
    'Nepali (macrolanguage)', 'Northern Sami', 'Norwegian', 'Norwegian Bokmål', 'Nyanja',
    'Oriya (macrolanguage)', 'Ossetian', 'Panjabi', 'Persian', 'Polish', 'Portuguese', 'Pushto',
    'Quechua', 'Romanian', 'Romansh', 'Russian', 'Samoan', 'Sango', 'Sanskrit', 'Sardinian',
    'Scottish Gaelic', 'Serbian', 'Serbo-Croatian', 'Sinhala', 'Slovak', 'Slovenian', 'Somali',
    'Southern Sotho', 'Spanish', 'Sundanese', 'Swahili (macrolanguage)', 'Swati', 'Swedish',
    'Tagalog', 'Tajik', 'Tamil', 'Tatar', 'Telugu', 'Thai', 'Tibetan', 'Tswana', 'Turkish', 'Twi',
    'Ukrainian', 'Urdu', 'Uzbek', 'Vietnamese', 'Welsh', 'Western Frisian', 'Wolof', 'Xhosa',
    'Yiddish', 'Yoruba', 'Zulu', 'unknown']

def apply_filters(df, options_genres, selected_years, options_actors):
    mask = pd.Series(True, index=df.index)
    
    if options_genres:
        selected_genre_columns = [f"genre_{genre}" for genre in options_genres]
        genre_mask = df[selected_genre_columns].sum(axis=1) == len(selected_genre_columns)
        mask &= genre_mask
    
    if options_actors:
        selected_actors = [actor.strip() for actor in options_actors.split(',')]
        actor_mask = df['contributors_actors'].str.contains(
            '|'.join(selected_actors), 
            case=False, 
            na=False
        )
        mask &= actor_mask
    
    if selected_years:
        year_mask = (df['startYear'] >= selected_years[0]) & \
                   (df['startYear'] <= selected_years[1])
        mask &= year_mask
    
    return df[mask]

def main():

    df_encoded = load_data('datasets_nettoye/df_encoded.csv')
    X_scaled, scaler = prepare_features(df_encoded)
    model = train_model(X_scaled.values)
    if st.button("🔄 Réinitialiser les filtres"):
        st.rerun() 
    st.title('🎬 Système de recommandation de films')
    show_movie_stats(df_encoded)
    
    # Films populaires :
    st.subheader("🌟 Films populaires")
    filtered_movies_votes = df_encoded[df_encoded['numVotes'] > 100000]
    filtered_top_movies = filtered_movies_votes.nlargest(5, ['averageRating', 'numVotes'])
    
    # Posters :
    cols = st.columns(5)
    for col, (_, movie) in zip(cols, filtered_top_movies.iterrows()):
        with col:
            if pd.notna(movie['poster_path']):
                img_url = f"https://image.tmdb.org/t/p/w500/{movie['poster_path']}"
                st.image(img_url, width=200)
            st.caption(f"{movie['primaryTitle']} ({int(movie['startYear'])})")
            st.caption(f"⭐ {movie['averageRating']}/10")
    
    st.markdown("---")
    
    # creation 2  colonnes pour les filtres principales 
    col1, col2 = st.columns(2)
    
    with col1:
        film_choisi = st.text_input("🎬 Rechercher un film :").strip().lower()
        options_genres = st.multiselect("🎭 Genres :", list_genres)
    
    with col2:
        options_actors = st.text_input("🌟 Acteurs (séparés par une virgule) :")
        selected_years = st.slider(
            "📅 Années :",
            min_value=int(df_encoded['startYear'].min()),
            max_value=int(df_encoded['startYear'].max()),
            value=(1995, 2024)
        )
    advanced_filters = create_sidebar()
    
    # bouton pour le recherche
    if st.button("🔍 Rechercher"):
        df_filtered = df_encoded[
            (df_encoded['averageRating'] >= advanced_filters['min_rating']) &
            (df_encoded['numVotes'] >= advanced_filters['min_votes']) &
            (df_encoded['runtimeMinutes'].between(*advanced_filters['duration_range']))
        ]
        
        if film_choisi:
            search_film = df_filtered[
                (df_filtered['originalTitle'].str.lower().str.contains(film_choisi, na=False)) |
                (df_filtered['primaryTitle'].str.lower().str.contains(film_choisi, na=False))
            ]
            
            if not search_film.empty:
                st.success(f"Film trouvé : {search_film.iloc[0]['primaryTitle']}")
                film_index = search_film.index[0]
                
                # Affichage du film choisi
                st.markdown("### 🎬 Film sélectionné :")
                afficher_film(search_film.iloc[0], list_genres, list_languages)
                
                # Obtenir et filtrer les recommandations
                distances, indices = model.kneighbors([X_scaled.iloc[film_index]])
                voisins = df_filtered.iloc[indices[0]]
                voisins_filtres = apply_filters(voisins, options_genres, selected_years, options_actors)
                
                if not voisins_filtres.empty:
                    st.markdown("### 🎯 Films similaires recommandés :")
                    for _, row in voisins_filtres.iterrows():
                        afficher_film(row, list_genres, list_languages)
                else:
                    st.warning("Aucun film similaire ne correspond à vos critères.")
            else:
                st.error("Film non trouvé dans la base de données.")
        else:
            # Recommandations base uniquement sur les filtres !
            filtered_results = apply_filters(df_filtered, options_genres, selected_years, options_actors)
            if not filtered_results.empty:
                st.markdown("### 📌 Films recommandés selon vos critères :")
                top_movies = filtered_results.nlargest(10, ['averageRating', 'numVotes'])
                for _, row in top_movies.iterrows():
                    afficher_film(row, list_genres, list_languages)
            else:
                st.warning("Aucun film ne correspond à vos critères.")
    
    # Partie supplementaire pour aide 
    with st.expander("ℹ️ Comment utiliser le système ?"):
        st.markdown("""
        1. **Recherche de film** : 
           - Entrez le titre d'un film que vous aimez
           - Le système trouvera des films similaires
        
        2. **Filtres principaux** :
           - Sélectionnez les genres qui vous intéressent
           - Entrez des noms d'acteurs
           - Choisissez une période
        
        3. **Filtres avancés** (dans la barre latérale) :
           - Définissez une note minimum
           - Choisissez un nombre minimum de votes
           - Sélectionnez une plage de durée
        
        4. **Utilisation** :
           - Cliquez sur 'Rechercher' pour voir les recommandations
           - Utilisez 'Réinitialiser les filtres' pour recommencer
           
        5. **Résultats** :
           - Les films sont affichés avec leurs informations détaillées
           - Cliquez sur les liens IMDB pour plus d'informations
        """)

if __name__ == "__main__":
    main()