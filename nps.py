import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
import joblib

df = pd.read_csv("International_Student.csv")
df.drop(['CaseId', 'CONSENT','Q6O', 'INFRASTRUCTUREEXPLAIN', 'SERVICEEXPLAIN', 'RESPONSIVEEXPLAIN',
        'RAPPORTEXPLAIN', 'SAFETYEXPLAIN','FOCUSEXPLAIN', 'CURRICULAEXPLAIN', 'INSTRUCTORSEXPLAIN',
        'COURSESEXPLAIN', 'SATISFACTIONEXPLAIN'], inplace=True, axis=1, errors='ignore')

data = df[['Q1', 'Q2', 'Q4', 'Q5','INFRASTRUCTURE_A1',
       'INFRASTRUCTURE_A2', 'INFRASTRUCTURE_A3', 'INFRASTRUCTURE_A4',
       'SERVICEABILITY_A1', 'SERVICEABILITY_A2', 'SERVICEABILITY_A3',
       'SERVICEABILITY_A4', 'SERVICEABILITY_A5', 'SERVICEABILITY_A6',
       'SERVICEABILITY_A7', 'SERVICEABILITY_A8', 'SERVICEABILITY_A9',
       'RESPONSIVENESS_A1', 'RESPONSIVENESS_A2', 'RESPONSIVENESS_A3',
       'RESPONSIVENESS_A4', 'RESPONSIVENESS_A5', 'RAPPORT_A1', 'RAPPORT_A2',
       'RAPPORT_A3', 'RAPPORT_A4', 'RAPPORT_A5', 'SAFETY_A1', 'SAFETY_A2',
       'SAFETY_A3', 'SAFETY_A4', 'SAFETY_A5', 'STUDENTFOCUS_A1',
       'STUDENTFOCUS_A2', 'STUDENTFOCUS_A3', 'STUDENTFOCUS_A4',
       'STUDENTFOCUS_A5', 'CURRICULA_A1', 'CURRICULA_A2', 'CURRICULA_A3',
       'CURRICULA_A4', 'INSTRUCTORS_A1', 'INSTRUCTORS_A2', 'INSTRUCTORS_A3',
       'INSTRUCTORS_A4', 'INSTRUCTORS_A5', 'COURSES_A1', 'COURSES_A2',
       'COURSES_A3', 'COURSES_A4', 'COURSES_A5', 'COURSES_A6',
       'SATISFACTION_A1', 'SATISFACTION_A2', 'SATISFACTION_A3',
       'SATISFACTION_A5', 'SATISFACTION_A7']]

data2 = data
num_columns_names = ['INFRASTRUCTURE_A1',
       'INFRASTRUCTURE_A2', 'INFRASTRUCTURE_A3', 'INFRASTRUCTURE_A4',
       'SERVICEABILITY_A1', 'SERVICEABILITY_A2', 'SERVICEABILITY_A3',
       'SERVICEABILITY_A4', 'SERVICEABILITY_A5', 'SERVICEABILITY_A6',
       'SERVICEABILITY_A7', 'SERVICEABILITY_A8', 'SERVICEABILITY_A9',
       'RESPONSIVENESS_A1', 'RESPONSIVENESS_A2', 'RESPONSIVENESS_A3',
       'RESPONSIVENESS_A4', 'RESPONSIVENESS_A5', 'RAPPORT_A1', 'RAPPORT_A2',
       'RAPPORT_A3', 'RAPPORT_A4', 'RAPPORT_A5', 'SAFETY_A1', 'SAFETY_A2',
       'SAFETY_A3', 'SAFETY_A4', 'SAFETY_A5', 'STUDENTFOCUS_A1',
       'STUDENTFOCUS_A2', 'STUDENTFOCUS_A3', 'STUDENTFOCUS_A4',
       'STUDENTFOCUS_A5', 'CURRICULA_A1', 'CURRICULA_A2', 'CURRICULA_A3',
       'CURRICULA_A4', 'INSTRUCTORS_A1', 'INSTRUCTORS_A2', 'INSTRUCTORS_A3',
       'INSTRUCTORS_A4', 'INSTRUCTORS_A5', 'COURSES_A1', 'COURSES_A2',
       'COURSES_A3', 'COURSES_A4', 'COURSES_A5', 'COURSES_A6',
       'SATISFACTION_A1', 'SATISFACTION_A2', 'SATISFACTION_A3',
       'SATISFACTION_A5', 'SATISFACTION_A7']
cat_columns = ['Q1', 'Q2', 'Q4', 'Q5']
lst_num_inputs = ['INFRASTRUCTURE_A1',
       'INFRASTRUCTURE_A2', 'INFRASTRUCTURE_A3', 'INFRASTRUCTURE_A4',
       'SERVICEABILITY_A1', 'SERVICEABILITY_A2', 'SERVICEABILITY_A3',
       'SERVICEABILITY_A4', 'SERVICEABILITY_A5', 'SERVICEABILITY_A6',
       'SERVICEABILITY_A7', 'SERVICEABILITY_A8', 'SERVICEABILITY_A9',
       'RESPONSIVENESS_A1', 'RESPONSIVENESS_A2', 'RESPONSIVENESS_A3',
       'RESPONSIVENESS_A4', 'RESPONSIVENESS_A5', 'RAPPORT_A1', 'RAPPORT_A2',
       'RAPPORT_A3', 'RAPPORT_A4', 'RAPPORT_A5', 'SAFETY_A1', 'SAFETY_A2',
       'SAFETY_A3', 'SAFETY_A4', 'SAFETY_A5', 'STUDENTFOCUS_A1',
       'STUDENTFOCUS_A2', 'STUDENTFOCUS_A3', 'STUDENTFOCUS_A4',
       'STUDENTFOCUS_A5', 'CURRICULA_A1', 'CURRICULA_A2', 'CURRICULA_A3',
       'CURRICULA_A4', 'INSTRUCTORS_A1', 'INSTRUCTORS_A2', 'INSTRUCTORS_A3',
       'INSTRUCTORS_A4', 'INSTRUCTORS_A5', 'COURSES_A1', 'COURSES_A2',
       'COURSES_A3', 'COURSES_A4', 'COURSES_A5', 'COURSES_A6']

lst_output = [ 'SATISFACTION_A1', 'SATISFACTION_A2', 'SATISFACTION_A3',
       'SATISFACTION_A5', 'SATISFACTION_A7']

for i in num_columns_names:
    data2[i] = data2[i].replace(' ', np.NaN, regex=True) # replace blank cell to np.NaN
    data2[i] = data2[i].astype('float64')
    data2[i] = data2[i].replace(np.NaN, round(data2[i].mean()), regex=True) # replace np.NaN cell to round mean
    
# identify outliers
outliers = data2[np.abs(data2['INFRASTRUCTURE_A1']-data2['INFRASTRUCTURE_A1'].mean()) > (3*data2['INFRASTRUCTURE_A1'].std())]

list_outliers_index = []

for i in num_columns_names:
    outliers = data2[np.abs(data2[i]-data2[i].mean()) > (3*data2[i].std())]
    temp = outliers.index.values.tolist()
    for j in temp:
        list_outliers_index.append(j)

import collections

counter=collections.Counter(list_outliers_index)

list_outliers_index_unique = set(list_outliers_index)

arr = [9,
 13,
 33,
 37,
 38,
 39,
 54,
 55,
 61,
 65,
 68,
 71,
 72,
 86,
 92,
 101,
 107,
 108,
 112,
 113,
 116,
 118,
 119,
 127,
 130,
 132,
 137,
 156,
 166,
 169,
 179,
 187,
 193,
 205]

update_data2 = data2
update_data2 = update_data2.drop(index=arr)
X = update_data2[['Q5','INFRASTRUCTURE_A1',
       'INFRASTRUCTURE_A2', 'INFRASTRUCTURE_A3', 'INFRASTRUCTURE_A4',
       'SERVICEABILITY_A1', 'SERVICEABILITY_A2', 'SERVICEABILITY_A3',
       'SERVICEABILITY_A4', 'SERVICEABILITY_A5', 'SERVICEABILITY_A6',
       'SERVICEABILITY_A7', 'SERVICEABILITY_A8', 'SERVICEABILITY_A9',
       'RESPONSIVENESS_A1', 'RESPONSIVENESS_A2', 'RESPONSIVENESS_A3',
       'RESPONSIVENESS_A4', 'RESPONSIVENESS_A5', 'RAPPORT_A1', 'RAPPORT_A2',
       'RAPPORT_A3', 'RAPPORT_A4', 'RAPPORT_A5', 'SAFETY_A1', 'SAFETY_A2',
       'SAFETY_A3', 'SAFETY_A4', 'SAFETY_A5', 'STUDENTFOCUS_A1',
       'STUDENTFOCUS_A2', 'STUDENTFOCUS_A3', 'STUDENTFOCUS_A4',
       'STUDENTFOCUS_A5', 'CURRICULA_A1', 'CURRICULA_A2', 'CURRICULA_A3',
       'CURRICULA_A4', 'INSTRUCTORS_A1', 'INSTRUCTORS_A2', 'INSTRUCTORS_A3',
       'INSTRUCTORS_A4', 'INSTRUCTORS_A5', 'COURSES_A1', 'COURSES_A2',
       'COURSES_A3', 'COURSES_A4', 'COURSES_A5', 'COURSES_A6']]

update_data2 = update_data2.reset_index()
update_data2['NPS'] = update_data2['SATISFACTION_A2']

for i in range(172):
    if update_data2['NPS'][i] < 4.1:
        update_data2['NPS'][i] = 'No'
    else:
        update_data2['NPS'][i] = 'Yes'

y = update_data2['NPS']
lst_lientuc_chosen_a = ['INFRASTRUCTURE_A1',
 'INFRASTRUCTURE_A2',
 'INFRASTRUCTURE_A3',
 'INFRASTRUCTURE_A4',
 'SERVICEABILITY_A1',
 'SERVICEABILITY_A2',
 'SERVICEABILITY_A3',
 'SERVICEABILITY_A4',
 'SERVICEABILITY_A5',
 'SERVICEABILITY_A6',
 'SERVICEABILITY_A7',
 'SERVICEABILITY_A8',
 'SERVICEABILITY_A9',
 'RESPONSIVENESS_A1',
 'RESPONSIVENESS_A2',
 'RESPONSIVENESS_A3',
 'RESPONSIVENESS_A4',
 'RESPONSIVENESS_A5',]

lst_lientuc_chosen_b = ['RAPPORT_A1',
 'RAPPORT_A2',
 'RAPPORT_A3',
 'RAPPORT_A4',
 'RAPPORT_A5',
 'SAFETY_A1',
 'SAFETY_A2',
 'SAFETY_A3',
 'SAFETY_A4',
 'SAFETY_A5',
 'STUDENTFOCUS_A1',
 'STUDENTFOCUS_A2',
 'STUDENTFOCUS_A3',
 'STUDENTFOCUS_A4',
 'STUDENTFOCUS_A5']

lst_lientuc_chosen_c = ['CURRICULA_A1',
 'CURRICULA_A2',
 'CURRICULA_A3',
 'CURRICULA_A4',
 'INSTRUCTORS_A1',
 'INSTRUCTORS_A2',
 'INSTRUCTORS_A3',
 'INSTRUCTORS_A4',
 'INSTRUCTORS_A5',
 'COURSES_A1',
 'COURSES_A2',
 'COURSES_A3',
 'COURSES_A4',
 'COURSES_A5',
 'COURSES_A6']

lst_lientuc_chosen = []
# Chuẩn hóa log normalization
# Lấy log các thuộc tính liên tục

for i in lst_lientuc_chosen_a:
  name_log = i + '_log'
  lst_lientuc_chosen.append(name_log)
  X[name_log] = np.log(X[i])

for i in lst_lientuc_chosen_b:
  name_log = i + '_log'
  lst_lientuc_chosen.append(name_log)
  X[name_log] = np.log(X[i])

for i in lst_lientuc_chosen_c:
  name_log = i + '_log'
  lst_lientuc_chosen.append(name_log)
  X[name_log] = np.log(X[i])

X_log = X[lst_lientuc_chosen]
X_log['Year'] = X['Q5']

X_log = X.drop(columns=['INFRASTRUCTURE_A1',
 'INFRASTRUCTURE_A2',
 'INFRASTRUCTURE_A3',
 'INFRASTRUCTURE_A4',
 'SERVICEABILITY_A1',
 'SERVICEABILITY_A2',
 'SERVICEABILITY_A3',
 'SERVICEABILITY_A4',
 'SERVICEABILITY_A5',
 'SERVICEABILITY_A6',
 'SERVICEABILITY_A7',
 'SERVICEABILITY_A8',
 'SERVICEABILITY_A9',
 'RESPONSIVENESS_A1',
 'RESPONSIVENESS_A2',
 'RESPONSIVENESS_A3',
 'RESPONSIVENESS_A4',
 'RESPONSIVENESS_A5',
 'RAPPORT_A1',
 'RAPPORT_A2',
 'RAPPORT_A3',
 'RAPPORT_A4',
 'RAPPORT_A5',
 'SAFETY_A1',
 'SAFETY_A2',
 'SAFETY_A3',
 'SAFETY_A4',
 'SAFETY_A5',
 'STUDENTFOCUS_A1',
 'STUDENTFOCUS_A2',
 'STUDENTFOCUS_A3',
 'STUDENTFOCUS_A4',
 'STUDENTFOCUS_A5',
 'CURRICULA_A1',
 'CURRICULA_A2',
 'CURRICULA_A3',
 'CURRICULA_A4',
 'INSTRUCTORS_A1',
 'INSTRUCTORS_A2',
 'INSTRUCTORS_A3',
 'INSTRUCTORS_A4',
 'INSTRUCTORS_A5',
 'COURSES_A1',
 'COURSES_A2',
 'COURSES_A3',
 'COURSES_A4',
 'COURSES_A5',
 'COURSES_A6'])

# build model

X_train, X_test, y_train, y_test = train_test_split(X_log, y, test_size=0.3)

model = joblib.load('CV_rfc_Model.pkl')
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


train_score = model.score(X_train_pca, y_train)
test_score = model.score(X_test_pca, y_test)
y_predict = model.predict(X_test_pca)
confusion = metrics.confusion_matrix(y_test, y_predict)
FN = confusion[1][0]
TN = confusion[0][0]
TP = confusion[1][1]
FP = confusion[0][1]

# ---Part 2: Show project result with Streamlit------------------------------------------
st.title('Data Science')
st.header('Net Promotion Score Predict Project')

# Tạo menu
menu = ['Overview', 'Build Projection', 'New Prediction']
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Overview':
# Hiển thị giới thiệu chung
    st.subheader("Overview")
    st.write("""
    #### Predict Net Promotion Score based on the survey of international student at a Canadian University
    - The data was preprocessed and unused features was eliminated.
    
    """)

elif choice == 'Build Projection':
# Xây dựng mô hình
    # Hiển thị dữ liệu
    st.subheader('Build Projection')
    st.write('### Data Processing')
    st.write('#### Show Data')
    st.table(update_data2.head(5))

    # Tham số đánh giá
    st.write('#### Build model and evaluation')
    st.write('Train set score: {}'.format(round(train_score,2)))
    st.write('Test set score: {}'.format(round(test_score,2)))
    st.write('Confussion matrix:')
    st.table(confusion)
    st.table(metrics.classification_report(y_test, y_predict))
    st.write(metrics.classification_report(y_test, y_predict))
    st.write('#### AUC: %.3f' %auc)

    # Trực quan hóa kết quả
    #st.write('#### Visualization')
    #fig, ax = plt.subplots()
    #ax.bar(['False Nagative', 'True Negative', 'True Positive', 'False Positive'], [FN, TN, TP, FP])
    #st.pyplot(fig)
    # roc curve
    st.write('ROC curve')
    #fig1, ax1 = plt.subplots()
    #ax1.plot([0,1],[0,1], linestyle='--')
    #ax1.plot(fpr, tpr, marker='.')
    #ax1.set_title('ROC curve')
    #ax1.set_xlabel('False Positive Rate')
    #ax1.set_ylabel('True Positive Rate')
    #st.pyplot(fig1)

elif choice == 'New Prediction':
    st.subheader("Make New Prediction")
    st.write("#### Input / Select Data")
    name = st.text_input('Name of Student:')

    #year
    Q5 = st.slider('Year',1,5,1)

    ### INFRASTRUCTURE'
    st.write("Infrastructure")
    INFRASTRUCTURE_1 = st.slider('INFRASTRUCTURE_A1',1,5,1) # 
    INFRASTRUCTURE_2 = st.slider('INFRASTRUCTURE_A2',1,5,1)
    INFRASTRUCTURE_3 = st.slider('INFRASTRUCTURE_A3',1,5,1)
    INFRASTRUCTURE_4 = st.slider('INFRASTRUCTURE_A4',1,5,1)

    ##SERVICE ABILITY
    st.write("Service Ability")
    SERVICEABILITY_1 = st.slider('SERVICEABILITY_A1',1,5,1)
    SERVICEABILITY_2 = st.slider('SERVICEABILITY_A2',1,5,1)
    SERVICEABILITY_3 = st.slider('SERVICEABILITY_A3',1,5,1)
    SERVICEABILITY_4 = st.slider('SERVICEABILITY_A4',1,5,1)
    SERVICEABILITY_5 = st.slider('SERVICEABILITY_A5',1,5,1)
    SERVICEABILITY_6 = st.slider('SERVICEABILITY_A6',1,5,1)
    SERVICEABILITY_7 = st.slider('SERVICEABILITY_A7',1,5,1)
    SERVICEABILITY_8 = st.slider('SERVICEABILITY_A8',1,5,1)
    SERVICEABILITY_9 = st.slider('SERVICEABILITY_A9',1,5,1)

    #Responsiveness
    RESPONSIVENESS_1 = st.slider('RESPONSIVENESS_A1',1,5,1)
    RESPONSIVENESS_2 = st.slider('RESPONSIVENESS_A2',1,5,1)
    RESPONSIVENESS_3 = st.slider('RESPONSIVENESS_A3',1,5,1)
    RESPONSIVENESS_4 = st.slider('RESPONSIVENESS_A4',1,5,1)
    RESPONSIVENESS_5 = st.slider('RESPONSIVENESS_A5',1,5,1)

    # Rapport
    RAPPORT_1 = st.slider('RAPPORT_A1',1,5,1)
    RAPPORT_2 = st.slider('RAPPORT_A2',1,5,1)
    RAPPORT_3 = st.slider('RAPPORT_A3',1,5,1)
    RAPPORT_4 = st.slider('RAPPORT_A4',1,5,1)
    RAPPORT_5 = st.slider('RAPPORT_A5',1,5,1)

    #safety
    SAFETY_1 = st.slider('SAFETY_A1',1,5,1)
    SAFETY_2 = st.slider('SAFETY_A2',1,5,1)
    SAFETY_3 = st.slider('SAFETY_A3',1,5,1)
    SAFETY_4 = st.slider('SAFETY_A4',1,5,1)
    SAFETY_5 = st.slider('SAFETY_A5',1,5,1)

    #Student Focus
    STUDENTFOCUS_1 = st.slider( 'STUDENTFOCUS_A1',1,5,1)
    STUDENTFOCUS_2 = st.slider( 'STUDENTFOCUS_A2',1,5,1)
    STUDENTFOCUS_3 = st.slider( 'STUDENTFOCUS_A3',1,5,1)
    STUDENTFOCUS_4 = st.slider( 'STUDENTFOCUS_A4',1,5,1)
    STUDENTFOCUS_5 = st.slider( 'STUDENTFOCUS_A5',1,5,1)

    # Curricula
    CURRICULA_1 = st.slider('CURRICULA_A1',1,5,1)
    CURRICULA_2 = st.slider('CURRICULA_A2',1,5,1)
    CURRICULA_3 = st.slider('CURRICULA_A3',1,5,1)
    CURRICULA_4 = st.slider('CURRICULA_A4',1,5,1)

    # Instructor
    INSTRUCTORS_1 = st.slider('INSTRUCTORS_A1',1,5,1)
    INSTRUCTORS_2 = st.slider('INSTRUCTORS_A2',1,5,1)
    INSTRUCTORS_3 = st.slider('INSTRUCTORS_A3',1,5,1)
    INSTRUCTORS_4 = st.slider('INSTRUCTORS_A4',1,5,1)
    INSTRUCTORS_5 = st.slider('INSTRUCTORS_A5',1,5,1)

    # courses
    COURSES_1 = st.slider('COURSES_A1',1,5,1)
    COURSES_2 = st.slider('COURSES_A2',1,5,1)
    COURSES_3 = st.slider('COURSES_A3',1,5,1)
    COURSES_4 = st.slider('COURSES_A4',1,5,1)
    COURSES_5 = st.slider('COURSES_A5',1,5,1)
    COURSES_6 = st.slider('COURSES_A6',1,5,1)


    # Make new prediction
    a = [INFRASTRUCTURE_1,INFRASTRUCTURE_2,INFRASTRUCTURE_3,INFRASTRUCTURE_4,
         SERVICEABILITY_1,SERVICEABILITY_2,SERVICEABILITY_3,SERVICEABILITY_4,SERVICEABILITY_5,
         SERVICEABILITY_6,SERVICEABILITY_7,SERVICEABILITY_8,SERVICEABILITY_9,
         RESPONSIVENESS_1,RESPONSIVENESS_2,RESPONSIVENESS_3,RESPONSIVENESS_4,RESPONSIVENESS_5,
         RAPPORT_1,RAPPORT_2,RAPPORT_3,RAPPORT_4,RAPPORT_5,
         SAFETY_1,SAFETY_2,SAFETY_3,SAFETY_4,SAFETY_5,
         STUDENTFOCUS_1,STUDENTFOCUS_2,STUDENTFOCUS_3,STUDENTFOCUS_4,STUDENTFOCUS_5,
         CURRICULA_1,CURRICULA_2,CURRICULA_3,CURRICULA_4,
         INSTRUCTORS_1,INSTRUCTORS_2,INSTRUCTORS_3,INSTRUCTORS_4,INSTRUCTORS_5,
         COURSES_1,COURSES_2,COURSES_3,COURSES_4,COURSES_5,COURSES_6]
      
    a_log = np.log(a)
    b = [Q5]


    #lst_lientuc_log = {}
    #for i in a:
      #j = 1
      #name_log = str(j) + '_log'
      #j = j+1
      #a[name_log] = np.log(a[i])
      #lst_lientuc_log.append(a[name_log])    

    new_data = scaler.transform([a_log])
    new_data_2 = Q5 + a_log
    prediction = model.predict(new_data_2)
    predict_probability = model.predict_proba(new_dat_2)

    if prediction[0] == 'Yes':
        	st.subheader('Student {} would have promote the school with a probability of {}%'.format(name, 
                                                    round(predict_probability[0][1]*100 , 2)))
    else:
	    st.subheader('Passenger {} would not have promote the school with a probability of {}%'.format(name, 
                                                    round(predict_probability[0][0]*100 , 2)))
