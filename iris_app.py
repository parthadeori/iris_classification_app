import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 🌸 Page setup
st.set_page_config(page_title="Iris Logistic Regression 🌿", layout="wide")

st.markdown("""
<style>
.stApp {
    background-color: #f9f9f9;
}
.main-title {
    text-align: center;
    color: #0077B6;
}
</style>
""", unsafe_allow_html=True)

# 🌼 Title and Welcome
st.markdown("<h1 class='main-title'>🌸 Iris Flower Classification App</h1>", unsafe_allow_html=True)
st.info("""
**Welcome!** 🌿  
This app predicts the species of an Iris flower using a **Logistic Regression model**.  
Use the sidebar to select example flowers or adjust feature sliders, then see the prediction and probabilities.
""")

# 📥 Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# 📊 Dataset Preview
st.subheader("📊 Dataset Preview")
n_rows = st.slider("Number of rows to preview:", 5, len(df), 10)
st.dataframe(df.head(n_rows))
with st.expander("🔍 View Full Dataset"):
    st.dataframe(df)

# 🌿 Sidebar: Example selector and custom inputs
st.sidebar.header("🌿 Select Example Flower or Custom Input")
example = st.sidebar.selectbox("Choose an example flower:",
                               ["Custom", "Setosa", "Versicolor", "Virginica"])

# Default slider values
sepal_length, sepal_width, petal_length, petal_width = 5.8, 3.0, 4.2, 1.3

# Set feature values based on example
if example == "Setosa":
    sepal_length, sepal_width, petal_length, petal_width = 5.0, 3.4, 1.5, 0.2
elif example == "Versicolor":
    sepal_length, sepal_width, petal_length, petal_width = 6.0, 2.8, 4.5, 1.4
elif example == "Virginica":
    sepal_length, sepal_width, petal_length, petal_width = 6.5, 3.0, 5.5, 2.0

# Sliders for custom input
if example == "Custom":
    sepal_length = st.sidebar.slider("Sepal length (cm)", 4.3, 7.9, 5.8)
    sepal_width = st.sidebar.slider("Sepal width (cm)", 2.0, 4.4, 3.0)
    petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 6.9, 4.2)
    petal_width = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 1.3)

# 🧠 Prepare data
X = df.iloc[:, :-1]
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧮 Train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 🎯 Model performance
st.subheader("📈 Model Performance")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# 🌼 Prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)
prediction = model.predict(input_data)
probs = model.predict_proba(input_data)

st.subheader("🌼 Prediction Result")
st.success(f"The predicted species is: **{prediction[0]}**")

# Probabilities bar chart
st.subheader("Prediction Probabilities")
prob_df = pd.DataFrame(probs, columns=iris.target_names)
st.bar_chart(prob_df.T)

# 💾 Download predictions
y_test_pred = model.predict(X_test)
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
csv = results.to_csv(index=False).encode('utf-8')
st.download_button(
    label="💾 Download Predictions as CSV",
    data=csv,
    file_name='iris_logistic_predictions.csv',
    mime='text/csv',
)

# 📊 Data Visualization
st.subheader("📊 Data Visualization")
selected_plot = st.selectbox(
    "Select Visualization Type",
    ["Pairplot", "Correlation Heatmap"]
)
if selected_plot == "Pairplot":
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)
elif selected_plot == "Correlation Heatmap":
    fig, ax = plt.subplots()
    sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# 📋 Feature importance / model insight
st.subheader("🧠 Model Feature Importance (Coefficients)")
coeff_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
st.bar_chart(coeff_df.set_index('Feature'))

# 📋 Classification report
with st.expander("📋 View Detailed Classification Report"):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)

# 🌸 Educational Notes with Fixed, Aligned Images (Latest Streamlit)
st.subheader("🌸 About Iris Flowers")
st.markdown("""
- **Setosa:** small flowers, short petals, widely separated  
- **Versicolor:** medium-sized flowers, longer petals  
- **Virginica:** largest flowers, long petals  
""")

col1, col2, col3 = st.columns(3)

col1.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/640px-Irissetosa1.jpg",
           caption="Setosa", use_container_width=True)
col2.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
           caption="Versicolor", use_container_width=True)
col3.image("https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg",
           caption="Virginica", use_container_width=True)
