import pickle
import streamlit as st
from utils import *
import base64

side_bg_ext = 'png'

st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open("plus.jpg", "rb").read()).decode()});
    }}
    </style>
    """,
    unsafe_allow_html=True,
    )


# sidebar for navigate
with st.sidebar:
    selected = st.selectbox(
        "HealthCare Misinformation Detection System using Machine Learning",
        ["Prediction", "About the System"],
    )

# Prediction Page
if selected == "Prediction":
    # page title
    st.title("HealthCare Misinformation Detection Predictor")
    st.subheader("Enter Tweet : ")
    tweet_data = st.text_area("", value="", height=280)
    select_notebook_file = st.selectbox(
        "Select the model versions to run over",
        [
            "covid_models_ML",
            "merged_models_ML",
            "userdefined DS",
            "Deep learning models",
        ],
    )
    if select_notebook_file == "Deep learning models":
        select_model = st.selectbox(
            "Select predefined or userdefined model",
            ["predefined_LSTM", "userdefined_LSTM", "GNN"],
        )
        if select_model == "predefined_LSTM":
            dl_model_path = "dl models/best_model predefined.h5"
            dl_tokenizer_path = "dl models/tokenizer_predefined.pickle"
        elif select_model == "userdefined_LSTM":
            dl_model_path = "dl models/best_model_userdefined.h5"
            dl_tokenizer_path = "dl models/tokenizer_predefined.pickle"
        elif select_model == "GNN":
            dl_tokenizer_path = "gnn_tokenizer.pkl"
            dl_model_path = "GNN model/cached_pmi_model.p"

    else:
        select_model = st.selectbox(
            "Select machine learning models to infer on",
            ["DT", "lgb_model", "LR", "passive_aggressive", "RFC"],
        )

    submit_button = st.button("Predict")

    if submit_button and select_notebook_file == "Deep learning models":
        if select_model == "GNN":
            prediction = predict_classes(
                tweet_data, "gnn_tokenizer.pkl", "GNN model/cached_pmi_model.p"
            )
            label_arr = ["Reliable", "Unreliable"]
            st.success("prediction done", icon="✅")
            st.title(f"We found this tweet to be {label_arr[prediction]}")
        else:
            prediction = predict_classes(tweet_data, dl_model_path, dl_tokenizer_path)
            st.success("Prediction done", icon="✅")
            st.title("We found this tweet to be " + str(prediction))

    elif submit_button and tweet_data:
        pred = make_pred(tweet_data, select_model, select_notebook_file)
        st.success("prediction done!", icon="✅")
        if pred[0] == 0 or pred[0] == 1:
            label_arr = ["Reliable", "Unreliable"]
            prediction = label_arr[pred[0]]
        else:
            prediction = pred[0]
        st.title("We found this tweet to be " + str(prediction))


# Prediction Page
if selected == "About the System":
    # page title
    st.title("HealthCare Misinformation Detection System")

    st.markdown(
        "*Recently, the use of social networks such as Facebook and Twitter has become an inseparable part of our daily lives. However, while people enjoy social networks, many deceptive activities such as fake news or rumors can mislead users into believing misinformation. Besides, spreading the massive amount of misinformation on social networks has become a global risk. Therefore, misinformation detection (MID) in social networks has gained a great deal of attention and is considered an emerging area of research interest. There is a need to develop automated solutions that can aid both experts and non-experts in differentiating between reliable and unreliable health information because healthcare is one of the most important and pressing topics in our society. Many malicious things about health are spread that are unreliable and people believe that to mislead that information.*"
    )
