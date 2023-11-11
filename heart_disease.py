# źródło danych [https://www.kaggle.com/c/titanic/](https://www.kaggle.com/c/titanic)
import os

import streamlit as st
import pickle
from datetime import datetime
import pandas as pd

startTime = datetime.now()
# import znanych nam bibliotek

import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


filename = "model.sv"

model = pickle.load(open(filename, 'rb'))
# otwieramy wcześniej wytrenowany model

sex_d = {0: "Mezczyzna", 1: "Kobieta", }
chest_pain_type_d = {0: "ATA", 1: "NAP", 2: "ASY", 3: "TA"}
resting_ecg_d = {0: "Normal", 1: "ST", 2: "LVH"}
exercise_angina_d = {0: "Nie", 1: "Tak"}
st_slope_d = {0: "Up", 1: "Flat", 2: "Down"}

filename2 = "DSP_2.csv"
base_data = pd.read_csv(filename2)


def main():
    st.set_page_config(page_title="Zadanie 4 - Piotr Trzos")
    overview = st.container()
    left, right = st.columns(2)
    prediction = st.container()

    st.image("https://img.freepik.com/free-photo/pleased-young-female-doctor-wearing-medical-robe-stethoscope-around-neck-standing-with-closed-posture_409827-254.jpg?size=626&ext=jpg&ga=GA1.1.1826414947.1699574400&semt=ais")

    with overview:
        st.title("Zadanie 4 - Piotr Trzos")

    with left:
        sex_radio = st.radio("Płeć", list(sex_d.keys()), format_func=lambda x: sex_d[x])

        chest_pain_type_radio = st.radio("Rodzaj bolu w klatce piersiowej", list(chest_pain_type_d.keys()),
                                         format_func=lambda x: chest_pain_type_d[x])

        resting_ecg_radio = st.radio("Spoczynkowe ekg", list(resting_ecg_d.keys()),
                                     format_func=lambda x: resting_ecg_d[x])

        exercise_angina_radio = st.radio("Angina Wysilkowa", list(exercise_angina_d.keys()),
                                         format_func=lambda x: exercise_angina_d[x])

        st_slope_radio = st.radio("Angina Wysilkowa", list(st_slope_d.keys()),
                                  format_func=lambda x: st_slope_d[x])

    with right:
        age_slider = st.slider("Wiek",
                               value=1,
                               min_value=int(base_data['Age'].min()),
                               max_value=int(base_data['Age'].max()))

        resting_bp_slider = st.slider("Spoczynkowe ciśnienie krwi",
                                      min_value=base_data['RestingBP'].min(),
                                      max_value=base_data['RestingBP'].max())

        cholesterol_slider = st.slider("Cholesterol",
                                       min_value=base_data['Cholesterol'].min(),
                                       max_value=base_data['Cholesterol'].max())

        fasting_bs_slider = st.slider("Cukier we krwi na czczo",
                                      min_value=int(base_data['FastingBS'].min()),
                                      max_value=int(base_data['FastingBS'].max()))

        max_hr_slider = st.slider("Maksymalne tętno",
                                  min_value=int(base_data['MaxHR'].min()),
                                  max_value=int(base_data['MaxHR'].max()))

        old_peak_slider = st.slider("Oldpeak",
                                    min_value=int(base_data['Oldpeak'].min()),
                                    max_value=int(base_data['Oldpeak'].max()))

    data = [[age_slider, sex_radio, chest_pain_type_radio, resting_bp_slider, cholesterol_slider, fasting_bs_slider,
             resting_ecg_radio, max_hr_slider, exercise_angina_radio, old_peak_slider, st_slope_radio]]
    survival = model.predict(data)
    s_confidence = model.predict_proba(data)

    with prediction:
        st.subheader("Czy taka osoba ma sznase na zawał?")
        st.subheader(("Tak" if survival[0] == 1 else "Nie"))
        st.write("Pewność predykcji {0:.2f} %".format(s_confidence[0][survival][0] * 100))


if __name__ == "__main__":
    main()
