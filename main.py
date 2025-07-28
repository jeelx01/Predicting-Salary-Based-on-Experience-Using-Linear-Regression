# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Read the CSV file
data = pd.read_csv("Salary_Data.csv")

# Step 2: Title and Dataset preview (optional)
st.title("Salary Prediction based on Experience")
st.write("## Dataset Preview")
st.dataframe(data.head())

# Step 3: Split data into input (X) and output (y)
X = data[['YearsExperience']]
y = data['Salary']

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Prediction input using Streamlit text input
st.write("## Enter years of experience to predict salary:")
experience = st.number_input("Years of Experience", min_value=0.0, step=0.5)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    if st.button("Predict Salary"):
        prediction = model.predict([[experience]])
        st.success(f"Predicted Salary for {experience} years experience: ₹{prediction[0]:.2f}")

# Step 7: Plotting
st.write("## Salary vs Experience Graph")
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Actual Data')
ax.plot(X, model.predict(X), color='red', label='Regression Line')
ax.set_xlabel("Years of Experience")
ax.set_ylabel("Salary")
ax.set_title("Salary vs Experience")
ax.legend()
st.pyplot(fig)
