import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score,mean_absolute_error,mean_squared_error
import streamlit as st

df = pd.read_csv("weight-height.csv")

df['Gender'] = df['Gender'].replace({'Male':1,'Female':0})

#df['Height'] = round(df['Height'],2)

#df['Weight'] = round(df['Weight'],2)

###

df['height_inch'] = df['Height']

#df['height_metre'] = round(df['height_inch']/39.37,2)
df['height_metre'] = df['height_inch']/39.37

df['weight_pound'] = df['Weight']

#df['weight_kg'] = round(df['weight_pound']/2.205,2)
df['weight_kg'] = df['weight_pound']/2.205

###

# The formula to calculate the BMI is: BMI = kg/m2:

df['bmi'] = round(df['weight_kg']/(df['height_metre']**2),2)

df = df.drop(columns=['Height','Weight','height_inch','weight_pound'])

# Splitting the dependant and independant features:

X = df.iloc[:,:3]
y = df.iloc[:,3:]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

### MODEL

linear = LinearRegression()

model = linear.fit(X_train,y_train)

y_pred = model.predict(X_test)







#####################################################

nav = st.sidebar.radio("Navigation Panel",['Home','Prediction'])
if nav == 'Home':
    st.write("**Hello. Welcome to the homepage!**")
elif nav == 'Prediction':
    st.write("**Hello. Welcome to the Prediction Page!**")
    st.write("**Enter your following details to find out your BMI Score and BMI Output**")


    gender = st.selectbox("Gender: Select 1 for Male & 0 for Female",df['Gender'].unique())
    #select_height = st.radio("Select the metric for Height. ",["Inches","Metres"])
    height = st.text_input("Enter your height: ")
    #select_weight = st.radio("Select the metric for Weight. ",["Pounds","Kilograms"])
    weight = st.text_input("Enter your weight: ")
    
    if st.button("Submit"):
        result = model.predict([[gender,height,weight]])

        st.write(f"**The BMI Score is {result}.**")

        if np.array(result) <= 18.4:
            st.write("**You are underweight.**")
        elif np.array(result) <= 24.9:
            st.write("**You are healthy.**")
        elif np.array(result) <= 29.9:
            st.write("**You are over weight.**")
        elif np.array(result) <= 34.9:
            st.write("**You are severely over weight.**")
        elif np.array(result) <= 39.9:
            st.write("**You are obese.**")
        else:
            st.write("**You are severely obese.**")

        #st.write(f"**The Root Mean Square Accuracy of the model is {np.sqrt(mean_squared_error(y_test,y_pred))}.**")

        
        #st.write(f"Your Height in Metres is {height}.\nYour Weight in Kilograms is {weight}.\nYour Calculated BMI score is {result}.\n.")





