import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
app=FastAPI()
ps=PorterStemmer()
def transform(text):
    y = []
    text = text.lower()  # lower case conversion
    text = nltk.word_tokenize(text)  # conversion string to list
    for i in text:
        if i.isalnum():  # removing specail characters
            y.append(i)
    # stopwords.words('english') = words used for sentence formation
    # string.punctuation = punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    # ps.stem() converts words to root form example playing to play
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

class ScoringItem(BaseModel):
    msg : str

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model_spam.pkl','rb'))

@app.post('/')

async def scoring_endpoint(item:ScoringItem):
    input_data = item.json()
    input_dictionary = json.loads(input_data)
    transformed_sms = transform(input_dictionary['msg'])
    vector_input = tfidf.transform([transformed_sms])
    result = model.predict(vector_input)[0]
    if result == 1:
        output = "spam"
    else:
        output = "notspam"
    return {"prediction":output}
