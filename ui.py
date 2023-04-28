import pickle
import streamlit as st


st.header("HealthCare Misinformation Detection System")

st.write(
    """
# Explore different classifier
Which one is the best ?
"""
)

dataset_name = st.selectbox(
    "Select Dataset : ",
    ("User Define Dataset", "Predefined Dataset of Covid", "Merge Dataset"),
)
st.write(dataset_name)

option_list_1 = ["Machine Learning Algorithm", "Deep Learning Algorithm"]
option_list_2 = [
    "Passive Aggressive",
    "Random Forest",
    "Decision Tree",
    "Logistic Regression",
    "Light BGM",
]
option_list_3 = ["LSTM", "GNN"]

selected_option_1 = st.selectbox(
    "Select an algorithm from the dropdown menu : ", option_list_1
)

if selected_option_1 == "Machine Learning Algorithm":
    st.write(selected_option_1)
    updated_option_list = option_list_2
elif selected_option_1 == "Deep Learning Algorithm":
    st.write(selected_option_1)
    updated_option_list = option_list_3
# else:
# updated_option_list = []

selected_option_2 = st.selectbox(
    "Select an algorithm / classifier : ", updated_option_list
)
st.write(selected_option_2)
