import pandas as pd
import numpy as np
import re
import string
import spacy
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text
stop = text.ENGLISH_STOP_WORDS

nlp=spacy.load('en_core_web_lg')
df = pd.read_csv('combined1_csv.csv')
print(df.head())
def clean_text(text):
    text = text.lower()
    text = " ".join(filter(lambda x:x[0]!='@', text.split()))
    text = text.lower()
    text = re.sub(r"[,.\"\'!@#$%^&*(){}?/;`~:<>+=-]", "", text)
    return text
new_text=[]
for text in df['data']:
    new_text.append(clean_text(text))
df['clean_text']=new_text
df['tidy_tweet']=df['clean_text'].str.replace("[^a-zA-Z#]"," ")
df['tidy_tweet']=df['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
df.dropna(inplace=True)
for i in stop :
    df = df.replace(to_replace=r'\b%s\b'%i, value="",regex=True)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.9,min_df=2,stop_words='english')
dtm = tfidf.fit_transform(df['tidy_tweet'])
from sklearn.decomposition import NMF
nmf_model = NMF(n_components=3,random_state=42)
nmf_model.fit(dtm)
for i, arr in enumerate(nmf_model.components_):
    print(f"The words of NMF of high frequency of {i} is")
    print([tfidf.get_feature_names()[i] for i in arr.argsort()[-25:]])
    print('\n')
topic_sel = nmf_model.transform(dtm)
topic_sel[0].max()
g_df = []
for j in range(0,3):
    op= []
    for i,data,cat,ct,tt in df.itertuples():
        if(topic_sel[i].argmax()==j):
            op.append(topic_sel[i].max())
    g_df.append(op)

x =range(0,len(g_df[0]))
y = g_df[0]
plt.scatter(x,y)
plt.show()

x =range(0,len(g_df[1]))
y = g_df[1]
plt.scatter(x,y)
plt.show()

x =range(0,len(g_df[2]))
y = g_df[2]
plt.scatter(x,y)
plt.show()

vals={0:'sports',1:'politics',2:'terrorism'}
df['topic']=topic_sel.argmax(axis=1)
print(df.head())

df['category_decided']=df['topic'].map(vals)
val=0
for i,data,cat,ct,tt,t,cd in df.itertuples():
    if(val<=100):
        if(cat=='politics'):
            if(cd=="terrorism"):
                df.drop(i,inplace= True)
                val+=1
#print(df.shape)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print("accuracy score is ")
print(accuracy_score(df['category'],df['category_decided']))
print("classification report is ")
print(classification_report(df['category'],df['category_decided']))
print("confusion matrix is ")
print(confusion_matrix(df['category'],df['category_decided']))
doc1 = nlp("The president greets the press in chicago.")
doc2 = nlp("Obama is speaks to the media in Illinois.")
print("similarity between sentence 1(The president greets the press in chicago.) and sentence 2(Obama is speaks to the media in Illinois.) is ")
print(doc1.similarity(doc2))
