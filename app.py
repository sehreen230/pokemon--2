import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os

st.set_page_config(page_title="Pokemon Master Dashboard", layout="wide", page_icon="⚡")

# Theming for seaborn
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

st.title("⚡ Pokemon Stats & Legendary Predictor ⚡")
st.markdown("Welcome to the ultimate Pokemon data exploration dashboard! Analyze stats, types, and test out our Machine Learning model that predicts if a custom Pokemon is **Legendary**!")

@st.cache_data
def load_data():
    if os.path.exists("Pokemon.csv"):
        return pd.read_csv("Pokemon.csv")
    
    desktop_path = os.path.expanduser("~/Desktop/pokemon_dataset")
    if os.path.exists(desktop_path):
        for f in os.listdir(desktop_path):
            if f.endswith('.csv'):
                return pd.read_csv(os.path.join(desktop_path, f))
    return None

df = load_data()

if df is None:
    st.error("Could not find Pokemon.csv in the current directory or ~/Desktop/pokemon_dataset. Please ensure the data exists!")
    st.stop()

stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Visualizations", "🔮 Legendary Predictor"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.write(f"**Shape:** {df.shape[0]} Pokemon, {df.shape[1]} features")
    st.write("Missing values summary:")
    # Filter only columns with missing values safely
    missing = df.isnull().sum()
    st.write(missing[missing > 0])

with tab2:
    st.header("Stat Distributions")
    fig1, axes1 = plt.subplots(2, 3, figsize=(14, 8))
    for ax, stat in zip(axes1.flatten(), stats):
        sns.histplot(df[stat], kde=True, ax=ax, color='steelblue')
        ax.set_title(stat)
    plt.tight_layout()
    st.pyplot(fig1)

    st.header("Type Frequency")
    if 'Type 1' in df.columns:
        type_counts = df['Type 1'].value_counts()
        fig2 = plt.figure(figsize=(12, 5))
        sns.barplot(x=type_counts.index, y=type_counts.values, palette='tab20')
        plt.xticks(rotation=45, ha='right')
        plt.title('Pokemon count by primary type')
        plt.ylabel('Count')
        plt.tight_layout()
        st.pyplot(fig2)

    st.header("Legendary vs Regular Pokemon")
    if 'Legendary' in df.columns:
        fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
        df['Legendary'].value_counts().plot.pie(
            ax=axes3[0], labels=['Regular', 'Legendary'],
            autopct='%1.1f%%', colors=['steelblue', 'gold'])
        axes3[0].set_title('Legendary ratio')

        sns.boxplot(data=df, x='Legendary', y='Total' if 'Total' in df.columns else stats[0], ax=axes3[1],
                    palette={False: 'steelblue', True: 'gold'})
        axes3[1].set_title('Total stats: Legendary vs Regular')
        plt.tight_layout()
        st.pyplot(fig3)
        
    st.header("Stat Correlation Matrix")
    fig4 = plt.figure(figsize=(9, 7))
    cols = stats[:]
    if 'Total' in df.columns:
        cols.append('Total')
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, linewidths=0.5)
    plt.title('Stat correlation matrix')
    plt.tight_layout()
    st.pyplot(fig4)

with tab3:
    st.header("Build a Pokemon & Predict if it's Legendary!")
    
    # Train model
    X = df[stats]
    y = df['Legendary'].astype(int)
    
    @st.cache_resource
    def train_model(X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model, X_test, y_test
        
    model, X_test, y_test = train_model(X, y)
    
    st.markdown("Adjust the sliders below to construct a custom Pokemon, and our Random Forest model will predict if it has what it takes to be Legendary!")
    
    col1, col2 = st.columns(2)
    with col1:
        hp = st.slider("HP", 1, 255, 106)
        attack = st.slider("Attack", 5, 190, 130)
        defense = st.slider("Defense", 5, 230, 130)
    with col2:
        sp_atk = st.slider("Sp. Atk", 10, 194, 90)
        sp_def = st.slider("Sp. Def", 20, 230, 154)
        speed = st.slider("Speed", 5, 180, 110)
        
    custom_df = pd.DataFrame([{
        'HP': hp, 'Attack': attack, 'Defense': defense,
        'Sp. Atk': sp_atk, 'Sp. Def': sp_def, 'Speed': speed
    }])
    
    prediction = model.predict(custom_df)[0]
    confidence = model.predict_proba(custom_df)[0].max() * 100
    
    st.subheader("Prediction Result:")
    if prediction == 1:
        st.success(f"🌟 **LEGENDARY!** ({confidence:.1f}% confident) 🌟")
    else:
        st.info(f"🟢 **Regular Pokemon** ({confidence:.1f}% confident)")
        
    st.markdown("---")
    st.subheader("Model Insights")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Feature Importance**")
        importance = pd.Series(model.feature_importances_, index=stats).sort_values(ascending=True)
        fig5 = plt.figure(figsize=(6, 4))
        colors = ['gold' if v == importance.max() else 'steelblue' for v in importance]
        plt.barh(importance.index, importance.values, color=colors)
        plt.tight_layout()
        st.pyplot(fig5)
        
    with col_b:
        st.markdown("**Confusion Matrix (Test Data)**")
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig6 = plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Regular', 'Legendary'],
                    yticklabels=['Regular', 'Legendary'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        st.pyplot(fig6)
