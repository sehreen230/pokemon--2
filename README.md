<img width="1440" height="600" alt="02_type_frequency" src="https://github.com/user-attachments/assets/9d24213b-ae39-4a0b-af7e-098ae2bd5032" />
# ⚡ Pokémon Dataset Analysis & ML Predictor

This project analyzes the stats and properties of 800 Pokémon and includes a Machine Learning model to predict whether a given Pokémon is "Legendary" or a regular Pokémon.

## 📊 Features
- **Data Visualization**: Automatically generates distribution plots, correlation heatmaps, and type frequency charts using Seaborn and Matplotlib.
- **Machine Learning**: Uses a `RandomForestClassifier` (from `scikit-learn`) to predict the "Legendary" status of a Pokémon based on its base stats (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed).
- **Custom Predictions**: Allows you to plug in stats for a custom made Pokémon to see if the AI classifies it as a Legendary!

## 📁 Files
- `pokemon.py`: The main Python script that loads the data, creates the graphical plots, trains the ML model, and outputs the confusion matrix.
- `Pokemon.csv`: The Kaggle dataset containing 800 Pokémon from the first 6 generations.

## ⚙️ Requirements
To run this project, you need the following Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

## 🚀 How to Run
Run the script from your terminal:
```bash
python3 pokemon.py
```

It will execute the full analysis quietly and save exactly 10 PNG images of the charts directly to your Desktop!
