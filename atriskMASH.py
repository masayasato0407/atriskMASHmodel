import streamlit as st
import pandas as pd
import joblib

#モデルの読み込み
@st.cache_resource
def load_model():
    return joblib.load('random_forest_model.pkl')

model = load_model()

# アプリのタイトル
st.title('Event Prediction App')

# 入力フォームの作成
st.header('Input Variables')

age = st.number_input('Age', min_value=18, max_value=100, value=50)

# DMとHTNをラジオボタンで選択
DM = 1 if st.radio('DM (Diabetes)', ['Absent', 'Present']) == 'Present' else 0
HTN = 1 if st.radio('HTN (Hypertension)', ['Absent', 'Present']) == 'Present' else 0

AST = st.number_input('AST', min_value=1, max_value=300, value=30)
GGT = st.number_input('γ-GTP', min_value=1, max_value=1000, value=30)
Plt = st.number_input('Plt', min_value=1.0, max_value=75.0, value=20.0)
INR = st.number_input('INR', min_value=0.50, max_value=3.00, value=1.00, step=0.01, format="%.2f")

# 入力データをデータフレームに変換
input_data = pd.DataFrame({
    'age': [age],
    'DM': [DM],
    'HTN': [HTN],
    'AST': [AST],
    'GGT': [GGT],
    'Plt': [Plt],
    'INR': [INR]
})

# 予測ボタン
if st.button('Predict'):
    # モデルの読み込みと予測
    model = load_model()
    probability = model.predict_proba(input_data)[0][1]
    
    # 結果の表示
    st.header('Prediction Result')
    st.write(f'Probability of having "at-risk MASH" in this patient = {probability:.2%}')
    
    # 入力データの表示
    st.subheader('Input Data Summary')
    st.write(input_data)