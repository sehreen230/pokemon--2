import kagglehub
import shutil
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 120

# ── Load data 
desktop_path = os.path.expanduser("~/Desktop/pokemon_dataset")

# Find the CSV file automatically
csv_file = None
for f in os.listdir(desktop_path):
    if f.endswith('.csv'):
        csv_file = os.path.join(desktop_path, f)
        break

if csv_file is None:
    print("ERROR: No CSV found in ~/Desktop/pokemon_dataset")
else:
    print(f"Found: {csv_file}")
    df = pd.read_csv(csv_file)

    # ── First look ─────────────────────────────────
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nFirst 5 rows:\n", df.head())

    # ── Stat distributions ─────────────────────────
    stats = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, stat in zip(axes.flatten(), stats):
        sns.histplot(df[stat], kde=True, ax=ax, color='steelblue')
        ax.set_title(stat)
    plt.suptitle('Distribution of Base Stats', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, '01_stat_distributions.png'))
    # plt.show()  # Commented out so execution doesn't pause
    print("Saved: 01_stat_distributions.png")

    # ── Type frequency ─────────────────────────────
    type_counts = df['Type 1'].value_counts()
    plt.figure(figsize=(12, 5))
    sns.barplot(x=type_counts.index, y=type_counts.values, palette='tab20')
    plt.xticks(rotation=45, ha='right')
    plt.title('Pokemon count by primary type')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, '02_type_frequency.png'))
    # plt.show()  # Commented out so execution doesn't pause
    print("Saved: 02_type_frequency.png")

    # ── Legendary vs Regular ───────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    df['Legendary'].value_counts().plot.pie(
        ax=axes[0], labels=['Regular', 'Legendary'],
        autopct='%1.1f%%', colors=['steelblue', 'gold'])
    axes[0].set_title('Legendary ratio')

    sns.boxplot(data=df, x='Legendary', y='Total', ax=axes[1],
                palette={False: 'steelblue', True: 'gold', 'False': 'steelblue', 'True': 'gold'})
    axes[1].set_title('Total stats: Legendary vs Regular')
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, '03_legendary_comparison.png'))
    # plt.show()  # Commented out so execution doesn't pause
    print("Saved: 03_legendary_comparison.png")

    # ── Correlation heatmap ────────────────────────
    plt.figure(figsize=(9, 7))
    corr = df[stats + ['Total']].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                square=True, linewidths=0.5)
    plt.title('Stat correlation matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(desktop_path, '04_correlation_heatmap.png'))
    # plt.show()  # Commented out so execution doesn't pause
    print("Saved: 04_correlation_heatmap.png")

    print("\nAll done! Check ~/Desktop/pokemon_dataset for your PNG files.")
# ── Legendary Predictor (ML Model) ─────────────────
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

print("\n=== Building Legendary Predictor ===")

# Prepare features and target
X = df[stats]
y = df['Legendary'].astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Results
y_pred = model.predict(X_test)
print("\n=== Model Performance ===")
print(classification_report(y_test, y_pred,
      target_names=['Regular', 'Legendary']))

# ── Feature importance chart ───────────────────────
importance = pd.Series(model.feature_importances_, index=stats)
importance = importance.sort_values(ascending=True)

plt.figure(figsize=(8, 5))
colors = ['gold' if v == importance.max() else 'steelblue' for v in importance]
plt.barh(importance.index, importance.values, color=colors)
plt.title('Which stats matter most for predicting Legendary?')
plt.xlabel('Feature importance')
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, '09_feature_importance.png'))
plt.show()
print("Saved: 09_feature_importance.png")

# ── Confusion matrix ───────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Regular', 'Legendary'],
            yticklabels=['Regular', 'Legendary'])
plt.title('Confusion matrix — Legendary predictor')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(desktop_path, '10_confusion_matrix.png'))
plt.show()
print("Saved: 10_confusion_matrix.png")

# ── Predict a custom Pokemon ───────────────────────
print("\n=== Test it yourself ===")
custom = pd.DataFrame([{
    'HP': 106, 'Attack': 130, 'Defense': 130,
    'Sp. Atk': 90, 'Sp. Def': 154, 'Speed': 110
}])
prediction = model.predict(custom)[0]
confidence = model.predict_proba(custom)[0].max() * 100
label = "LEGENDARY" if prediction == 1 else "Regular"
print(f"Custom Pokemon prediction: {label} ({confidence:.1f}% confident)")

print("\n============================")
print("PROJECT COMPLETE!")
print("============================")
print(f"Total charts saved: 10 PNGs")
print(f"Cleaned dataset:    pokemon_cleaned.csv")
print(f"Location:           {desktop_path}")