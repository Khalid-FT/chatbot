
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from methodes import chatbot_response

model = load_model('training/chatbot_model.h5')
import json
intents = json.loads(open('training/intents.json').read())
words = pickle.load(open('training/words.pkl','rb'))
classes = pickle.load(open('training/classes.pkl','rb'))


from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route("/get", methods=["GET", "POST"])
def response():
        query_msg = request.args.get("msg") 
        ans =chatbot_response(query_msg, model , classes , words , intents)
        return str(ans) 
     
if __name__ == '__main__':
   app.run()
                