import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score 

def main():
    
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Add your mushrooms edible or poisonous🍄")
    st.sidebar.markdown("Add your mushrooms edible or poisonous🍄")

    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv("F:/Machine Learning/MLCONCEPTS/Webapps/ML/mushrooms.csv")
        label = LabelEncoder()
        for col in data.columns:
            data[col] = label.fit_transform(data[col])
        return data

    @st.cache(persist = True)
    def split(df):
        y = df.type
        x = df.drop(columns=['type'])
        x_train, x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3, random_state = 0)
        return x_train,x_test,y_train,y_test

    def plot_matrics(metrics_list):
        if "Confusion Matrix" in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()

        if "ROC Curve" in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test, display_labels = class_names)
            st.pyplot()

        if "Precision-Recall Curve" in mtrics_list:
            st.subheader("Precision-Recall Curve")
            plot_confusion_matrix(model, x_test, y_test, display_labels = class_names)
            st.pyplot()
    

       
    df = load_data() 
    x_train,x_test,y_train,y_test = split(df) 
    class_names = ['edible','poisonous']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier",("Support Vector Machine (SVM)","LOgistic Regression","Random Forest"))


    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step = 0.01,key = "C")
        kernel = st.sidebar.radio("kernal",("rbf","linear"), key = "kernal")
        gamma = st.sidebar.radio("Gamma (Kernal Coefficient",("scale","auto"),key = "gamma")

        metrics = st.sidebar.multiselect("Choose to plot",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key = "classify"):
            st.subheader("Support Vector Machine(SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test,y_pred, labels=class_names).round(2))
            plot_matrics(metrics)


    if classifier == 'LOgistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)",0.01,10.0,step = 0.01,key = "C_LR")
        max_iter = st.sidebar.slider("Max no. of iterations",100,500, key = "max_iter")
       
        metrics = st.sidebar.multiselect("Choose to plot",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key = "classify"):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test,y_pred, labels=class_names).round(2))
            plot_matrics(metrics)        

    


    if classifier == 'Random_Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("no of trees in forest",100,5000,step=10,key=n_estimators)
        max_depth = st.sidebar.number_input("The max depth of tree",1,20,step=1,key=max_depth)
        bootstrap = st.sidebar.radio("Bootstrap samples while building a tree",("True","False"), key="bootstrap")
        metrics = st.sidebar.multiselect("Choose to plot",('Confusion Matrix','ROC Curve','Precision-Recall Curve'))

        if st.sidebar.button("Classify",key = "classify"):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap=bootstrap)
            model.fit(x_train,y_train)
            accuracy = model.score(x_test,y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ",accuracy.round(2))
            st.write("Precision: ", precision_score(y_test,y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test,y_pred, labels=class_names).round(2))
            plot_matrics(metrics)        

    


if __name__ == '__main__':
    main()


