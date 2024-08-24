import streamlit as st
from predict_page import show_predict_page

# from explore_page import show_explore_page
from Dashboard import dashboard

page = st.sidebar.selectbox(
    "Prediction Page or Dashboard Page", ("Prediction Page", "Dashboard Page")
)

if page == "Prediction Page":
    show_predict_page()
else:
    dashboard()


# if __name__ == "__main__":
#     show_predict_page()
#     show_explore_page()
