#%%load package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import shap
import sklearn
import joblib

#%%不提示warning信息
st.set_option('deprecation.showPyplotGlobalUse', False)

#%%set title
st.set_page_config(page_title='Machine learning algorithm to construct a preoperative clinical-radiomics prediction model for the ki-67 proliferation index in patients with pituitary adenomas')
st.title('Machine learning algorithm to construct a preoperative clinical-radiomics prediction model for the ki-67 proliferation index in patients with pituitary adenomas')

#%%set varixgbles selection
st.sidebar.markdown('## Varixgbles')

prl = st.sidebar.slider("Prolactin (ng/mL)",0.00,260.00,value=40.00, step=0.01)
age = st.sidebar.slider("Age (year)",10,80,value = 40,step = 1)
waveletLLL_glcm = st.sidebar.slider("waveletLLL_glcm",-2.00,3.00,value = 1.00,step = 0.01)
size = st.sidebar.selectbox('Tumor Size',('Microadenoma(diameter＜1.0cm)','Macroadenoma(diameter≥1.0cm)',
                                          'Giant adenoma(diameter≥3.0cm)'),index=1)
cs = st.sidebar.selectbox('Cavernous Sinus Invasion',('No','Yes'),index=1)
logsigma_firstorder = st.sidebar.slider("logsigma_firstorder", -2.00, 3.00, value=1.00, step = 0.01)
original_gldm = st.sidebar.slider("original_gldm", -2.00, 7.00, value=1.00, step = 0.01)
waveletLHL_glcm = st.sidebar.slider("waveletLHL_glcm", -5.00, 5.00, value=1.00, step = 0.01)


#分割符号
st.sidebar.markdown('#  ')
st.sidebar.markdown('#  ')
st.sidebar.markdown('##### All rights reserved') 
st.sidebar.markdown('##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
#传入数据
map = {'Microadenoma(diameter＜1.0cm)':1,'Macroadenoma(diameter≥1.0cm)':2,'Giant adenoma(diameter≥3.0cm)':3,'No':0,'Yes':1}
size =map[size]
cs = map[cs]
# 数据读取，特征标注
#%%load model
xgb_model = joblib.load('xgb_model.pkl')

#%%load data
hp_train = pd.read_csv('combine.csv')
features = ['prl',
            'age',
            'waveletLLL_glcm',
            'size',
            'cs',
            'logsigma_firstorder',
            'original_gldm',
            'waveletLHL_glcm']
target = 'ki67'
y = np.array(hp_train[target])
sp = 0.5

is_t = (xgb_model.predict_proba(np.array([[prl,age,waveletLLL_glcm,size,cs,logsigma_firstorder,original_gldm,waveletLHL_glcm]]))[0][1])> sp
prob = (xgb_model.predict_proba(np.array([[prl,age,waveletLLL_glcm,size,cs,logsigma_firstorder,original_gldm,waveletLHL_glcm]]))[0][1])*1000//1/10
    

if is_t:
    result = 'High Risk Group'
else:
    result = 'Low Risk Group'
if st.button('Predict'):
    st.markdown('## Result:  '+str(result))
    if result == '  Low Risk Group':
        st.balloons()
    st.markdown('## Probxgbility of High Risk group:  '+str(prob)+'%')
    #%%cbind users data
    col_names = features
    X_last = pd.DataFrame(np.array([[prl,age,waveletLLL_glcm,size,cs,logsigma_firstorder,original_gldm,waveletLHL_glcm]]))
    X_last.columns = col_names
    X_raw = hp_train[features]
    X = pd.concat([X_raw,X_last],ignore_index=True)
    if is_t:
        y_last = 1
    else:
        y_last = 0  
    y_raw = (np.array(hp_train[target]))
    y = np.append(y_raw,y_last)
    y = pd.DataFrame(y)
    model = xgb_model
    #%%calculate shap values
    sns.set()
    explainer = shap.Explainer(model, X)
    shap_values = explainer.shap_values(X)
    a = len(X)-1
    #%%SHAP Force logit plot
    st.subheader('SHAP Force logit plot of XGB model')
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    force_plot = shap.force_plot(explainer.expected_value,
                    shap_values[a, :], 
                    X.iloc[a, :], 
                    figsize=(25, 3),
                    # link = "logit",
                    matplotlib=True,
                    out_names = "Output value")
    st.pyplot(force_plot)
    #%%SHAP Water PLOT
    st.subheader('SHAP Water plot of XGB model')
    shap_values = explainer(X) # 传入特征矩阵X，计算SHAP值
    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    waterfall_plot = shap.plots.waterfall(shap_values[a,:])
    st.pyplot(waterfall_plot)
    #%%ConfusionMatrix 
    st.subheader('Confusion Matrix of XGB model')
    xgb_prob = xgb_model.predict(X)
    cm = confusion_matrix(y, xgb_prob)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low risk', 'High risk'])
    sns.set_style("white")
    disp.plot(cmap='RdPu')
    plt.title("Confusion Matrix of XGB model")
    disp1 = plt.show()
    st.pyplot(disp1)

