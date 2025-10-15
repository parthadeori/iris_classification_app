import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import plotly.express as px
import plotly.figure_factory as ff
import altair as alt

# 🌸 Page setup
st.set_page_config(page_title="Iris Flower Classification 🌿", layout="wide")

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

# 🎯 Interactive Model Performance
st.subheader("📈 Model Performance (Interactive)")

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
st.success(f"**Accuracy:** {accuracy:.2f} ✅")
st.info("Accuracy tells us **how often the model predicts the correct species overall**.")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
labels = ['Setosa', 'Versicolor', 'Virginica']

# Create hover text for each cell
hover_text = []
for i in range(cm.shape[0]):
    hover_text.append([])
    for j in range(cm.shape[1]):
        hover_text[i].append(f"Actual: {labels[i]}<br>Predicted: {labels[j]}<br>Count: {cm[i,j]}")

# Create interactive heatmap
fig = ff.create_annotated_heatmap(
    z=cm,
    x=labels,
    y=labels,
    annotation_text=cm,
    colorscale='Blues',
    hoverinfo='text',
    text=hover_text
)

fig.update_layout(
    title_text="Confusion Matrix 🧐",
    xaxis_title="Predicted Species",
    yaxis_title="Actual Species",
    yaxis=dict(autorange='reversed')  # reverse y-axis so top-left is first row
)

st.plotly_chart(fig, width='stretch')

# Beginner-friendly explanation
st.info("""
**How to read this confusion matrix:**  

- **Rows** = Actual species  
- **Columns** = Predicted species  

✅ Cells on the diagonal = correct predictions  
❌ Off-diagonal cells = misclassifications  

Hover over each cell to see **exact counts and which species were confused**.  
Higher numbers on the diagonal → model is doing well  
Higher numbers off-diagonal → model is confusing some species
""")

# 🌼 Prediction
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                          columns=iris.feature_names)
prediction = model.predict(input_data)
probs = model.predict_proba(input_data)

st.subheader("🌼 Prediction Result")
st.success(f"The predicted species is: **{prediction[0]}**")

st.subheader("Prediction Probabilities 🌿")
st.info("""
This bar chart shows **how confident the model is** about each species:

- **Red:** Setosa  
- **Blue:** Versicolor  
- **Green:** Virginica  

The taller the bar, the more confident the model is that your flower belongs to that species.  
If bars are close in height, the model is less certain.
""")

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

# 📊 Interactive Data Visualization (Beginner-friendly)
st.subheader("📊 Data Visualization")

st.info("""
These interactive graphs help you **explore the dataset**:

- **Scatter matrix (Pairplot):** Each dot is a flower.  
  - Hover over a dot to see its features and species.  
  - Colors show species (Setosa, Versicolor, Virginica).  
  - Helps you see patterns — e.g., Setosa petals are usually shorter than others.

- **Correlation Heatmap:** Shows how strongly features are related.  
  - Darker squares = stronger correlation.  
  - Hover over a square to see the correlation value.
""")

selected_plot = st.selectbox(
    "Select Visualization Type",
    ["Interactive Pairplot", "Interactive Correlation Heatmap"]
)

if selected_plot == "Interactive Pairplot":
    fig = px.scatter_matrix(
        df,
        dimensions=iris.feature_names,
        color="species",
        symbol="species",
        title="Interactive Pairplot of Iris Dataset",
        hover_data=["species"]
    )
    st.plotly_chart(fig, width='stretch')

elif selected_plot == "Interactive Correlation Heatmap":
    corr_matrix = df.iloc[:, :-1].corr()
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.columns),
        colorscale='Viridis',
        showscale=True,
        hoverinfo="z"
    )
    fig.update_layout(title="Interactive Correlation Heatmap")
    st.plotly_chart(fig, width='stretch')

# 📋 Beginner-friendly Interactive Feature Importance
st.subheader("🧠 Feature Importance / Model Insight")

st.info("""
This chart shows **how each feature affects the model's prediction**:

- Positive values mean **higher values of this feature increase the likelihood** of the predicted species.  
- Negative values mean **higher values decrease the likelihood**.  
- Taller bars = more influence on the prediction.  

For example, in Iris classification:
- **Petal length and petal width** usually have the strongest effect.
""")

# Prepare DataFrame
coeff_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
})

# Create color column for positive/negative
coeff_df['Color'] = coeff_df['Coefficient'].apply(lambda x: 'green' if x > 0 else 'red')

# Altair horizontal bar chart
feature_chart = alt.Chart(coeff_df).mark_bar().encode(
    x=alt.X('Coefficient:Q', title='Coefficient Value'),
    y=alt.Y('Feature:N', sort='-x', title='Feature'),
    color=alt.Color('Color:N', scale=None, legend=None),
    tooltip=['Feature', 'Coefficient']
).properties(width=700, height=300)

st.altair_chart(feature_chart, use_container_width=True)

# 📋 Beginner-friendly Visual Classification Report with Support
st.subheader("📋 Visual Model Evaluation (with Support)")

st.info("""
This chart shows **how well the model predicts each species** and **how many flowers of each species were in the test set**.  

- **Green bars:** Model is doing great ✅  
- **Yellow bars:** Model is okay ⚠️  
- **Red bars:** Model needs improvement ❌  

Metrics:  
- **Precision:** When the model predicts a species, how often is it correct?  
- **Recall:** How many actual flowers of this species did the model catch?  
- **F1-score:** A balance between precision and recall (higher is better).  
- **Support:** Number of flowers of that species in the test set (shown on top of the bars).
""")

# Get classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().reset_index()

# Keep only species rows
species_df = report_df[~report_df['index'].str.contains('avg|accuracy')].copy()

# Rename columns
species_df.rename(columns={
    'index': 'Species',
    'precision': 'Precision',
    'recall': 'Recall',
    'f1-score': 'F1 Score',
    'support': 'Support'
}, inplace=True)

# Melt for plotting
plot_df = species_df.melt(
    id_vars=['Species', 'Support'],
    value_vars=['Precision', 'Recall', 'F1 Score'],
    var_name='Metric',
    value_name='Score'
)

fig = px.bar(
    plot_df,
    x='Species',
    y='Score',
    color='Score',
    text='Support',  # Show support as text on bars
    facet_col='Metric',
    color_continuous_scale='RdYlGn',  # Red → Yellow → Green
    range_color=[0, 1],
    height=400,
    title='Visual Classification Report with Support'
)
fig.update_traces(textposition='outside', texttemplate='%{text}')  # Show support on top
fig.update_layout(showlegend=False)

st.plotly_chart(fig, width='stretch')

# 🌸 Educational Notes with Your Specified Images
st.subheader("🌸 About Iris Flowers")
st.markdown("""
- **Setosa:** small flowers, short petals, widely separated  
- **Versicolor:** medium-sized flowers, longer petals  
- **Virginica:** largest flowers, long petals  
""")

col1, col2, col3 = st.columns(3)
col1.image("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Irissetosa1.jpg/640px-Irissetosa1.jpg",
           caption="Setosa", use_container_width=True)
col2.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/Iris_versicolor_4.jpg/640px-Iris_versicolor_4.jpg",
           caption="Versicolor", use_container_width=True)
col3.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Iris_virginica_NRCS-4x3.jpg/640px-Iris_virginica_NRCS-4x3.jpg",
           caption="Virginica", use_container_width=True)

# 🌿 Simple Interactive Flower Prediction
st.subheader("🌿 Try Your Own Flower")

st.info("""
**How to read this graph:**  

- Each **dot** represents a real Iris flower from the dataset:  
  - **Blue:** Setosa  
  - **Red:** Versicolor  
  - **Green:** Virginica  

- The **black X** represents the flower you selected using the sliders.  

- The plot shows **where your flower sits compared to real flowers** based on **petal length and petal width**.  
- This helps beginners **visualize why the model predicts its species**.
""")


# Sliders
petal_length_input = st.slider("Petal Length (cm)", float(df['petal length (cm)'].min()),
                               float(df['petal length (cm)'].max()), 4.0)
petal_width_input = st.slider("Petal Width (cm)", float(df['petal width (cm)'].min()),
                              float(df['petal width (cm)'].max()), 1.2)

# Predict
input_point = pd.DataFrame([[petal_length_input, petal_width_input]],
                           columns=['petal length (cm)', 'petal width (cm)'])
model_2d = LogisticRegression()
model_2d.fit(df[['petal length (cm)', 'petal width (cm)']], df['species'])
predicted_species = model_2d.predict(input_point)[0]
pred_prob = model_2d.predict_proba(input_point)[0]

st.success(f"Predicted Species: **{predicted_species}**")
st.write("Prediction Probabilities:")
st.dataframe(pd.DataFrame([pred_prob], columns=model_2d.classes_))

# Simple scatter of real flowers + user selection
fig, ax = plt.subplots()
sns.scatterplot(data=df, x='petal length (cm)', y='petal width (cm)',
                hue='species', palette='Set1', s=80, ax=ax)
ax.scatter(petal_length_input, petal_width_input, color='black', s=150, marker='X', label='Your Input')
ax.set_title("Your Flower vs Real Flowers")
st.pyplot(fig)

# Dynamic beginner-friendly messages below the scatter plot
if predicted_species == 'setosa':
    st.info("Your flower (black X) is in the red cluster → looks like a **Setosa**! 🌸")
elif predicted_species == 'versicolor':
    st.info("Your flower (black X) is in the blue cluster → looks like a **Versicolor**! 🌺")
else:
    st.info("Your flower (black X) is in the green cluster → looks like a **Virginica**! 🌼")

# Comment on prediction confidence
max_prob = pred_prob.max()
if max_prob < 0.6:
    st.warning("The model is **not very confident** about this prediction. Try adjusting the sliders!")
else:
    st.success("The model is **confident** about this prediction!")
