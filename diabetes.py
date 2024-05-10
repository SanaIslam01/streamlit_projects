# # import all important libraries
# import pandas as pd 
# import numpy as np 
# import seaborn as sns 
# import matplotlib.pyplot as plt 

# # Read DataSet
# dataset = pd.read_csv('diabetes.csv')
# dataset.head()

# # use shape function to check the total number rows and columns in our dataset
# dataset.shape

# # use info function to about the know about the datatypes 
# dataset.info()

# # find missing values in the form of percentage
# (dataset.isnull().sum()/dataset.shape[0])*100

# # distribution of diabteas & normal people
# dataset['Outcome'].value_counts()

# #checking the duplicate values
# dataset.duplicated()

# #checking the correlation by using heatmap
# sns.heatmap(dataset.corr(), vmin=-1, vmax=1,
# annot=True,cmap="rocket_r")
# plt.show()

# sns.countplot(x='Outcome',data= dataset,palette = 'flare')
# plt.show()

# # Assuming "size" column is missing or has non-numeric values
# sns.scatterplot(
#     data=dataset, x="Age", y="BloodPressure", hue="Outcome",size="Outcome",
#     sizes=(100, 50),hue_norm=(0, 2), legend="full"
# )

# # Assuming "size" column is missing or has non-numeric values
# sns.scatterplot(
#     data=dataset, x="BMI", y="Glucose", hue="Outcome",size="Outcome",
#     sizes=(100, 50),hue_norm=(0, 2),palette="husl"
# )

# #checking 0 values in dataset

# (dataset[dataset.columns] == 0).sum()

# # Replacing 0 values with median

# for i in ["Glucose","BMI","Insulin","BloodPressure"]:
#     dataset[i].replace({0:dataset[i].median()},inplace = True)

# #again checking 0 values in dataset
# (dataset[dataset.columns] == 0).sum()

# # checking the outliers
# # sns.boxplot(data=dataset)
# sns.boxplot(data=dataset[["Age", "BMI", "BloodPressure"]])
# plt.show()

# # checking the outliers
# # sns.boxplot(data=dataset)
# sns.boxplot(data=dataset[["Glucose", "Insulin"]])
# plt.show()

# # Code-2 of removing outlier
# def outlier_treatment():
#     l = ["BMI","Glucose","SkinThickness","Age","BloodPressure","Insulin","Pregnancies","DiabetesPedigreeFunction"]
#     for i in l:
#         x = np.quantile(dataset[i],[0.25,0.75])
#         iqr = x[1]-x[0]
#         uw = x[1]+1.5*iqr
#         lw = x[0]-1.5*iqr
#         dataset[i]  = np.where(dataset[i]>uw,uw,(np.where(dataset[i]<lw,lw,dataset[i])))
        
# outlier_treatment()

# # checking the outliers
# # sns.boxplot(data=dataset)
# sns.boxplot(data=dataset[["Age", "BMI", "BloodPressure","Insulin"]])
# plt.show()

# dataset.shape

# # Doing Feature Scaling by using standarization method

# sns.distplot(dataset["Age"], label="Age", kde=1)  # Column 1 ka histogram plot
# sns.distplot(dataset["BMI"], label="BMI", kde=1)  # Column 2 ka histogram plot
# sns.distplot(dataset["BloodPressure"], label="BloodPressure", kde=1)  # Column 3 ka histogram plot
# # Plot titles and labels
# plt.title("Distribution of Age,BMI and BloodPressure")
# plt.xlabel("Value")
# plt.ylabel("Frequency")

# # Legend 
# plt.legend()
# # show plot
# plt.show()


# sns.distplot(dataset["Insulin"], label="Insulin", kde=1)  # Column 1 ka histogram plot
# # Plottitle and labels
# plt.title("Distribution Insulin")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# # Legend 
# plt.legend()
# # show plot
# plt.show()

# # checking the statistical summary of dataset
# dataset.describe()

# #separating target variable

# X = dataset[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
# y = dataset['Outcome']

# # x = dataset.drop(columns=['Outcome'])
# # y = dataset['Outcome']

# X.head()

# # import libraries and split data 
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# #Scaling the Data
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# # fit the model
# from sklearn.svm import SVC
# svc_model = SVC(kernel='linear', C=1.0) # default regularization
# svc_model.fit(x_train,y_train)
# y_pred = svc_model.predict(x_test)

# # Model fitness score checking
# print("score for test data =",svc_model.score(x_test, y_test))

# print("score for training data = ",svc_model.score(x_train, y_train))

# # creating the report
# from sklearn.metrics import accuracy_score, classification_report
# report = classification_report(y_test, y_pred)
# print(report)

# # Now Evaluation Using confusion_matrix
# from sklearn.metrics import confusion_matrix,accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)

# sns.heatmap(cm, annot=True)
# plt.show()

# sns.heatmap(cm/np.sum(cm), annot=True, 
#             fmt='.2%', cmap='Blues')
# plt.show()

# labels = ['True Neg','False Pos','False Neg','True Pos']
# labels = np.asarray(labels).reshape(2,2)
# sns.heatmap(cm, annot=labels, fmt='', cmap='Blues')
# plt.show()

# acc1 = accuracy_score(y_test, y_pred)
# print(f"Accuracy score: {acc1}")

# dataset.columns

# # step no 6 - user input
# import streamlit as st
# st.title("Welcome to my app")
# st.balloons()
# col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
# with col1:
#     p_pregnancies = st.text_input(float(input("Enter Pregnancies: ")))
# with col2:
#     p_glucose = st.text_input(float(input("Enter glucose: ")))
# with col3:
#     p_bloodPressure = st.text_input(float(input("Enter bp: ")))
# with col4:
#     p_skinThickness = st.text_input(float(input("Enter SkinThickness: ")))
# with col5:
#     p_insulin = st.text_input(float(input("Enter Insulin: ")))
# with col6:
#     p_bmi = st.text_input(float(input("Enter bmi: ")))
# with col7:
#     p_diabetesPedigreeFunction = st.text_input(float(input("Enter DiabetesPedigreeFunction: ")))
# with col8:
#     p_age = st.text_input(float(input("Enter age: ")))



 
# user_input = np.array([[p_pregnancies,p_glucose,p_bloodPressure,p_skinThickness,p_insulin,p_bmi,p_diabetesPedigreeFunction,p_age]])
# prediction = svc_model.predict(user_input)
# if prediction[0] == 0:
#     print("The patient don't have diabteas. ")
# else:
#     print("yes the patient has diabtes:")

















import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to read the dataset
def read_dataset():
    return pd.read_csv('diabetes.csv')

# Function to handle outliers
def outlier_treatment(dataset):
    l = ["BMI","Glucose","SkinThickness","Age","BloodPressure","Insulin","Pregnancies","DiabetesPedigreeFunction"]
    for i in l:
        x = np.quantile(dataset[i],[0.25,0.75])
        iqr = x[1]-x[0]
        uw = x[1]+1.5*iqr
        lw = x[0]-1.5*iqr
        dataset[i]  = np.where(dataset[i]>uw,uw,(np.where(dataset[i]<lw,lw,dataset[i])))

# Function to preprocess dataset
def preprocess_data(dataset):
    # Handling missing values
    for i in ["Glucose","BMI","Insulin","BloodPressure"]:
        dataset[i].replace({0:dataset[i].median()}, inplace=True)
    
    # Handling outliers
    outlier_treatment(dataset)
    
    # Separating features and target
    X = dataset[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
    y = dataset['Outcome']
    
    # Splitting the dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scaling the data
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)
    
    return x_train, x_test, y_train, y_test

# Function to build and train the model
def build_model(x_train, y_train):
    svc_model = SVC(kernel='linear', C=1.0) # default regularization
    svc_model.fit(x_train, y_train)
    return svc_model

# Function to display classification report and confusion matrix
def display_evaluation(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    st.text("Classification Report:")
    st.text(report)
    
    cm = confusion_matrix(y_test, y_pred)
    st.text("Confusion Matrix:")
    st.write(cm)
    
    st.text("Confusion Matrix (Normalized):")
    st.write(cm / np.sum(cm))

# Function to make prediction
def make_prediction(model, user_input):
    prediction = model.predict(user_input)
    return prediction

# Function to set background image
def set_background_image(image_url):
    page_bg = f'''
    <style>
    .stApp {{
        background: url("{image_url}");
        background-size: cover;
    }} 
    </style>
    '''
    st.markdown(page_bg, unsafe_allow_html=True)

# Set background image
set_background_image("https://cdn.pixabay.com/photo/2020/06/19/22/33/wormhole-5319067_960_720.jpg")

# Read dataset
dataset = read_dataset()

# Preprocess data
x_train, x_test, y_train, y_test = preprocess_data(dataset)

# Build and train the model
svc_model = build_model(x_train, y_train)

# Title of the app
st.title("Diabetes Prediction App")

# User input section
st.header("Enter Patient Details:")
col1, col2= st.columns(2)
col3, col4= st.columns(2)
col5, col6= st.columns(2)
col7, col8= st.columns(2)
with col1:
    p_pregnancies = st.number_input("Enter Pregnancies", value=0.0, step=1.0)
with col2:
    p_glucose = st.number_input("Enter Glucose", value=0.0, step=1.0)
with col3:
    p_bloodPressure = st.number_input("Enter Blood Pressure", value=0.0, step=1.0)
with col4:
    p_skinThickness = st.number_input("Enter Skin Thickness", value=0.0, step=1.0)
with col5:
    p_insulin = st.number_input("Enter Insulin", value=0.0, step=1.0)
with col6:
    p_bmi = st.number_input("Enter BMI", value=0.0, step=1.0)
with col7:
    p_diabetesPedigreeFunction = st.number_input("Enter Diabetes Pedigree Function", value=0.0, step=0.01)
with col8:
    p_age = st.number_input("Enter Age", value=0.0, step=1.0)

# user_input = np.array([[p_pregnancies,p_glucose,p_bloodPressure,p_skinThickness,p_insulin,p_bmi,p_diabetesPedigreeFunction,p_age]])
# prediction = make_prediction(svc_model, user_input)

# Button to submit the form
button = st.button("Done")

if button:
    # Display the submitted data
    # st.markdown(f"""
    # p_pregnancies: {p_pregnancies}
    # p_glucose: {p_glucose}
    # p_bloodPressure: {p_bloodPressure}
    # p_skinThickness: {p_skinThickness}
    # p_insulin: {p_insulin}
    # p_bmi: {p_bmi}
    # p_diabetesPedigreeFunction: {p_diabetesPedigreeFunction}
    # p_age: {p_age}
    # """)

 user_input = np.array([[p_pregnancies,p_glucose,p_bloodPressure,p_skinThickness,p_insulin,p_bmi,p_diabetesPedigreeFunction,p_age]])
 prediction = make_prediction(svc_model, user_input)


 if prediction[0] == 0:
    st.write("The patient doesn't have diabetes.")
 else:
    st.write("Yes, the patient has diabetes.")

# # Evaluation section
# st.header("Model Evaluation:")
# st.text("Score for Test Data:")
# st.write(svc_model.score(x_test, y_test))

# st.text("Score for Training Data:")
# st.write(svc_model.score(x_train, y_train))

# # Display evaluation metrics
# y_pred = svc_model.predict(x_test)
# display_evaluation(y_test, y_pred)
