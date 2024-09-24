import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# タイトル
st.title("Explainable AI App")
st.write("kmeansによるクラスタリングを決定木で近似して説明可能性を向上")
st.markdown("<br>", unsafe_allow_html=True)

csv_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if csv_file is not None:
        # CSVファイルをデータフレームに読み込む
        df = pd.read_csv(csv_file, header=None)
        st.markdown("<br>", unsafe_allow_html=True)
        
        #入力されたファイルを表示
        st.write("あなたが入力したファイル")
        st.write(df)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 使用する特徴量の列を選択
        feature_column = st.multiselect("クラスタリングに用いる特徴量の列を選択してください", df.columns)
        
        if feature_column:
            # 選択された特徴量のデータフレームを作成
            selected_df = df[feature_column]

            # 数値に変換する必要がある列を確認
            label_encoders = {}
            for column in selected_df.columns:
                if isinstance(selected_df[column].iloc[1], str) and not selected_df[column].iloc[1].isdigit():
                    le = LabelEncoder()
                    selected_df[column].iloc[1:] = le.fit_transform(selected_df[column].iloc[1:])
                    label_encoders[column] = le

            # 数値変換の対応表を表示
            for column, le in label_encoders.items():
                st.write(f"'{column}'の数値変換対応表:")
                st.write(dict(zip(le.classes_, le.transform(le.classes_))))
                        

            # クラスタ数の入力
            n_clusters = st.number_input("クラスタ数を入力してください", min_value=2, max_value=10, value=3, step=1)
            st.markdown("<br>", unsafe_allow_html=True)
            button1 = st.button("クラスタリング開始")
            
            if button1:
                # KMeansクラスタリングの実行
                
                data_array = selected_df.iloc[1:].to_numpy()
                kmeans_model = KMeans(n_clusters=n_clusters, random_state=0).fit(data_array)
                labels = kmeans_model.labels_

                # クラスタリング結果をデータフレームに追加
                clustered_df = selected_df.copy()
                clustered_df['Cluster'] = labels

                # クラスタリング結果を表示
                st.write("クラスタリング結果")
                st.write(clustered_df)         
                
    