import streamlit as st
import pickle
import pandas as pd

st.header('Прогнозирование заболеваний сердца')
lc, cc, rc = st.columns(3)
with lc:
    age = st.slider('Введите возраст:', key='age')
    gender = st.radio('Выберете пол:', ('М', 'Ж'), key='gender')
    height = st.slider('Введите рост в см:', 125, 225, key='height')
    weight = st.slider('Введите вес в кг:', 40, 300, key='weight')
    ap_hi = st.slider('Верхняя граница АД:', 125, 225, key='ap_hi')
    ap_lo = st.slider('Нижняя граница АД:', 40, 300, key='ap_lo')
    cholesterol = st.selectbox('Выберете уровень холестерина:', [1, 2, 3], key='cholesterol')
    gluc = st.selectbox('Уровень глюкозы:', [1, 2, 3], key='gluc')
    smoke = st.checkbox('Курит?', key='smoke')
    alco = st.checkbox('Употребляет алкоголь?', key='alco')
    active = st.checkbox('Занимается спортом?', key='active')
with rc:
    st.subheader('Данные пациента:')
    st.write('Возраст:', age)
    st.write('Пол:', gender)
    st.write('Рост:', height, 'см')
    st.write('Вес:', weight, 'кг')
    st.write('Верхняя граница АД:', ap_hi, 'мм рт.ст.')
    st.write('Нижняя граница АД:', ap_lo, 'мм рт.ст.')
    st.write('Уровень холестерина:', cholesterol)
    st.write('Уровень глюкозы:', gluc)
    st.write('Курит:', smoke)
    st.write('Употребляет алкоголь:', alco)
    st.write('Занимается спортом:', active)

bmi = weight / (height / 100)**2
with rc:
    st.write('Индекс массы тела:', bmi)

def load():
    with open('./models/model.pcl', 'rb') as fid:
        return pickle.load(fid)

model = load()

df = pd.DataFrame([[age, gender, ap_hi, ap_lo, cholesterol,
                    gluc, smoke, alco, active, bmi]])
df.columns = ['age', 'gender', 'ap_hi', 'ap_lo', 'cholesterol',
              'gluc', 'smoke', 'alco', 'active', 'bmi']

y_pred = model.predict_proba(df)[:, 1]

st.subheader('Вероятность возникновения заболеваний сердца:')
st.write(y_pred[0])