from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the machine learning model
with open('model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

#country encoding
with open('decoded_country.pkl', 'rb') as f:
    decode = pickle.load(f)

#scaling
with open('scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.form
    print(data)
    # Extract the relevant features from the input data
    year = data['year']
    schizophrenia = data['scp']
    bipolar = data['bipolardisorder']
    eating_disorders = data['etd']
    anxiety_disorders = data['anxietydisorder']
    drug_use_disorders = data['dud']
    depressive_disorders = data['depressivedisorder']
    alcohol_use_disorders = data['aud']
    Country = data['country']
    
    #preprocessing
    Country = decode[Country]
    
    #dataframe
    ip_df = pd.DataFrame(data=[
        [Country, year, schizophrenia, bipolar, eating_disorders, anxiety_disorders, drug_use_disorders, depressive_disorders, alcohol_use_disorders]
        ], 
        columns=['Year','Schizophrenia(%)','Bipolar disorder(%)','Eating disorders(%)',
               'Anxiety disorders(%)','Drug use disorders(%)','Depressive disorders(%)','Alcohol use disorders(%)','Entity_encoded'])
    
    #scaling
    ip_sc = scaler.transform(ip_df)
    
    
    
    # Perform the prediction using the loaded model
    prediction = model_loaded.predict(ip_sc)
    
    def categorize_mental_fitness(rate):
        if rate <= 4:
            return 'Low'
        elif rate <= 9:
            return 'Moderate'
        else:
            return 'High'

    def get_category_details(category):
        if category == 'Low':
            return "Your predicted mental fitness rate is low. It's important to prioritize your mental well-being and seek professional help. Consider consulting with a mental health professional or therapist for guidance and support. Focus on self-care activities, engage in hobbies that bring you joy, and maintain a strong support network."
        elif category == 'Moderate':
            return "Your predicted mental fitness rate is moderate. Continue to pay attention to your mental well-being and consider engaging in activities that promote positive mental health. Explore mindfulness exercises, practice self-reflection, and seek support from loved ones or support groups. Consider consulting with a mental health professional if needed."
        else:
            return "Congratulations! Your predicted mental fitness rate is high. This indicates a positive mental well-being. Continue to prioritize your mental health by engaging in activities that promote well-being, maintaining healthy habits, and nurturing strong relationships. Remember to practice self-care and be mindful of any changes that may occur."

    # Example usage
    mental_fitness_rate = prediction[0]
    category = categorize_mental_fitness(mental_fitness_rate)
    details = get_category_details(category)
        
    # Prepare the response data
    response = {
        'prediction': prediction[0],
        'details': details
    }
    
    return render_template("result.html", data=response, round=round)

# import requests
# from flask import Flask, jsonify

# app = Flask(__name__)

# @app.route('/form-submission', methods=['POST'])
# def handle_submission():
#     if request.method == 'POST':
#         # Parse the JSON data from the request
#         data = request.json
#         print(data)
#         # Access the submitted form fields
#         email = data['email']
#         description = data['name']
        
#         # Perform any necessary processing or validation
        
#         # Return a 200 status code to indicate successful processing
#         return '', 200
    
#     # Return a 405 status code for other request methods
#     return '', 405

if __name__ == '__main__':
    app.run(debug=False)
