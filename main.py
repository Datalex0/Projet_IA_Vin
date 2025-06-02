
import streamlit as st
from routes import routes

#Configuration des dimensions & affichage de la page
st.set_page_config(page_title="Appli Pipeline", 
                   page_icon="📊", 
                   layout='wide')


pages = st.navigation(routes)
pages.run()