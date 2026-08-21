# ============================================================
# TASK 1: IRIS FLOWER CLASSIFICATION
# Internship Machine Learning Project
# ============================================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="wide"
)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("🌸 Iris Flower Classification")
st.write(
    "A Machine Learning application that classifies Iris flowers "
    "into Setosa, Versicolor, and Virginica based on flower measurements."
)

st.divider()

# ------------------------------------------------------------
# LOAD IRIS DATASET
# ------------------------------------------------------------

@st.cache_data
def load_data():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    df["species"] = [
        iris.target_names[target]
        for target in iris.target
    ]

    return df, iris


df, iris = load_data()

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("⚙️ Model Settings")

test_size = st.sidebar.slider(
    "Test Data Percentage",
    min_value=10,
    max_value=40,
    value=20,
    step=5
)

random_state = st.sidebar.number_input(
    "Random State",
    min_value=0,
    max_value=100,
    value=42
)

# ------------------------------------------------------------
# DATASET INFORMATION
# ------------------------------------------------------------

st.header("📊 Dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples", len(df))

with col2:
    st.metric("Features", 4)

with col3:
    st.metric("Classes", 3)

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

# ------------------------------------------------------------
# DATASET STATISTICS
# ------------------------------------------------------------

with st.expander("📈 View Dataset Statistics"):

    st.write("### Statistical Summary")
    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    st.write("### Class Distribution")
    st.bar_chart(df["species"].value_counts())

# ------------------------------------------------------------
# PREPARE DATA
# ------------------------------------------------------------

X = df[
    [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ]
]

y = iris.target

# ------------------------------------------------------------
# TRAIN / TEST SPLIT
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=int(random_state),
    stratify=y
)

# ------------------------------------------------------------
# FEATURE SCALING
# ------------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------
# TRAIN MACHINE LEARNING MODEL
# ------------------------------------------------------------

model = LogisticRegression(
    max_iter=200
)

model.fit(
    X_train_scaled,
    y_train
)

# ------------------------------------------------------------
# MODEL PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(X_test_scaled)

# ------------------------------------------------------------
# MODEL EVALUATION
# ------------------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

st.divider()

st.header("🤖 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy * 100:.2f}%"
    )

with col2:
    st.metric(
        "Training Samples",
        len(X_train)
    )

with col3:
    st.metric(
        "Testing Samples",
        len(X_test)
    )

# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(3),
    use_container_width=True
)

# ------------------------------------------------------------
# CONFUSION MATRIX
# ------------------------------------------------------------

st.subheader("🔲 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred
)

fig, ax = plt.subplots()

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

disp.plot(
    ax=ax,
    values_format="d"
)

ax.set_title("Iris Classification Confusion Matrix")

st.pyplot(fig)

# ------------------------------------------------------------
# FEATURE IMPORTANCE / MODEL COEFFICIENTS
# ------------------------------------------------------------

st.subheader("📌 Model Coefficients")

coefficient_df = pd.DataFrame(
    model.coef_,
    columns=X.columns,
    index=iris.target_names
)

st.dataframe(
    coefficient_df.round(3),
    use_container_width=True
)

# ------------------------------------------------------------
# FLOWER PREDICTION
# ------------------------------------------------------------

st.divider()

st.header("🌺 Predict Iris Flower")

st.write(
    "Enter the flower measurements below to predict its species."
)

col1, col2 = st.columns(2)

with col1:

    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=5.1,
        step=0.1
    )

    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=3.5,
        step=0.1
    )

with col2:

    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=10.0,
        value=1.4,
        step=0.1
    )

    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=10.0,
        value=0.2,
        step=0.1
    )

# ------------------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------------------

if st.button(
    "🔮 Predict Flower Species",
    use_container_width=True
):

    input_data = pd.DataFrame(
        [[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]],
        columns=[
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(
        input_scaled
    )[0]

    probabilities = model.predict_proba(
        input_scaled
    )[0]

    predicted_species = iris.target_names[
        prediction
    ]

    confidence = probabilities[
        prediction
    ] * 100

    st.success(
        f"🌸 Predicted Species: **{predicted_species.capitalize()}**"
    )

    st.info(
        f"Prediction Confidence: **{confidence:.2f}%**"
    )

    # Probability table

    probability_df = pd.DataFrame({
        "Species": [
            species.capitalize()
            for species in iris.target_names
        ],
        "Probability (%)": [
            probability * 100
            for probability in probabilities
        ]
    })

    st.subheader("Prediction Probabilities")

    st.dataframe(
        probability_df.round(2),
        use_container_width=True
    )

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.divider()

st.caption(
    "Iris Flower Classification | Machine Learning Internship Task 1"
)