from flask import Flask, request, jsonify
from my_processor import evaluate
from my_function import get_class
from dotenv import load_dotenv
import os
load_dotenv()
PORT = os.getenv('PORT') 
app = Flask(__name__)


@app.route('/', methods=['GET' , 'POST'])
def home():
    if request.method=='GET':
        return 'Welcome to the Kavida server'
    elif request.method=='POST':
        input_text = request.json['input_text']
        predicted_label , label_probabilities = evaluate(input_text)
        res = {}
        res['predicted_label'] = get_class(predicted_label)
        res['label_probabilties'] = label_probabilities
        return jsonify(res)
    else:
        return 'Method Not defined'

if __name__ == '__main__':
    app.run(debug=True , host='0.0.0.0' , port=PORT)