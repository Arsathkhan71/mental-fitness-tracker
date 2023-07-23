from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import pickle
import pandas as pd
import secrets

app = Flask(__name__)

# Generate a secret key
secret_key = secrets.token_hex(16)
app.secret_key = secret_key

# Load the machine learning model
with open('model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

#country encoding
with open('decoded_country.pkl', 'rb') as f:
    decode = pickle.load(f)

#scaling
with open('scaling.pkl', 'rb') as f:
    scaler = pickle.load(f)
    

    
def recommendations(predicted_mental_fitness):
    # More activity data with additional recommendations
    activity_data = {
    'low': {
        'Meditation': [
            {'title': 'Guided Meditation', 'msg': "Meditation is a practice in which an individual uses a technique – such as mindfulness, or focusing the mind on a particular object, thought, or activity – to train attention and awareness, and achieve a mentally clear and emotionally calm and stable state."},
            {'title': 'Breathing Exercises', 'msg': "Breathing exercises are a way to decrease stress, ease anxiety, improve sleep and boost your mood — plus address COPD and blood pressure problems."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Mindfulness Exercises': [
            {'title': 'Mindful Eating', 'msg': "Mindful eating is a practice of paying full attention to the process of eating and drinking, both inside and outside the body. It involves all the senses and is based on the idea of being present in the moment."},
            {'title': 'Nature Observation', 'msg': "Nature observation involves mindfully observing and connecting with the natural world, such as watching wildlife, clouds, or trees."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Self-Care Routine': [
            {'title': 'Practice Gratitude', 'msg': "Practicing gratitude involves reflecting on and expressing appreciation for the positive aspects of life."},
            {'title': 'Listen to Music',  'msg': "Listening to music mindfully can be a form of relaxation and a way to connect with emotions."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Relaxation Exercises': [
            {'title': 'Deep Breathing',  'msg': "Deep breathing exercises help calm the mind and relax the body, reducing stress and promoting overall well-being."},
            {'title': 'Visualization',  'msg': "Visualization involves using the imagination to create mental images that promote relaxation and positive thinking."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Physical Exercises': [
            {'title': 'Yoga Class',  'msg': "Yoga is a mind-body practice that combines physical postures, breathing techniques, and meditation or relaxation."},
            {'title': 'Dancing', 'msg': "Dancing is a fun way to stay active and improve physical fitness while expressing creativity."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Gratitude Journaling': [
            {'title': 'Write a gratitude letter to someone',  'msg': "Writing a gratitude letter is a way to express appreciation and positive feelings to someone you care about."},
            {'title': 'Create a gratitude collage',  'msg': "Creating a gratitude collage involves making a visual representation of the things you are grateful for."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Creative Expression': [
            {'title': 'Writing in a journal',  'msg': "Writing in a journal can be a therapeutic way to express thoughts and emotions."},
            {'title': 'Playing a musical instrument',  'msg': "Playing a musical instrument can be a creative and fulfilling way to relax and engage the mind."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Social Activities': [
            {'title': 'Volunteer for a cause you care about', 'msg': "Volunteering is a way to contribute to the community and make a positive impact."},
            {'title': 'Host a small gathering',   'msg': "Hosting a small gathering with friends or family can foster social connections and create a sense of belonging."},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
    },
    'moderate': {
        'Meditation': [
            {'title': 'Guided3 Meditation',  'msg': "Moderate Level Meditation Activity Description"},
            {'title': 'Breathing Exercises',  'msg': "Moderate Level Breathing Exercise Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Mindfulness Exercises': [
            {'title': 'Mindful Eating','msg': "Moderate Level Mindful Eating Activity Description"},
            {'title': 'Nature Observation',  'msg': "Moderate Level Nature Observation Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Self-Care Routine': [
            {'title': 'Practice Gratitude',  'msg': "Moderate Level Practice Gratitude Activity Description"},
            {'title': 'Listen to Music', 'msg': "Moderate Level Listen to Music Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Relaxation Exercises': [
            {'title': 'Deep Breathing',  'msg': "Moderate Level Deep Breathing Activity Description"},
            {'title': 'Visualization', 'msg': "Moderate Level Visualization Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Physical Exercises': [
            {'title': 'Yoga Class',  'msg': "Moderate Level Yoga Class Activity Description"},
            {'title': 'Dancing', 'msg': "Moderate Level Dancing Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Gratitude Journaling': [
            {'title': 'Write a gratitude letter to someone', 'msg': "Moderate Level Write a Gratitude Letter Activity Description"},
            {'title': 'Create a gratitude collage',  'msg': "Moderate Level Create a Gratitude Collage Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Creative Expression': [
            {'title': 'Writing in a journal',  'msg': "Moderate Level Writing in a Journal Activity Description"},
            {'title': 'Playing a musical instrument',  'msg': "Moderate Level Playing a Musical Instrument Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Social Activities': [
            {'title': 'Volunteer for a cause you care about',  'msg': "Moderate Level Volunteer for a Cause Activity Description"},
            {'title': 'Host a small gathering','msg': "Moderate Level Host a Small Gathering Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
    },
    'high': {
        'Meditation': [
            {'title': 'Guided2 Meditation',  'msg': "High Level Meditation Activity Description"},
            {'title': 'Breathing Exercises',  'msg': "High Level Breathing Exercise Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Mindfulness Exercises': [
            {'title': 'Mindful Walking', 'image': 'https://example.com/mindful_walking_image.jpg', 'msg': "High Level Mindful Walking Activity Description"},
            {'title': 'Mindful Journaling', 'image': 'https://example.com/mindful_journaling_image.jpg', 'msg': "High Level Mindful Journaling Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Self-Care Routine': [
            {'title': 'Connect with Friends', 'image': 'https://example.com/connect_with_friends_image.jpg', 'msg': "High Level Connect with Friends Activity Description"},
            {'title': 'Practice a Hobby', 'image': 'https://example.com/practice_hobby_image.jpg', 'msg': "High Level Practice a Hobby Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Relaxation Exercises': [
            {'title': 'Tai Chi', 'image': 'https://example.com/tai_chi_image.jpg', 'msg': "High Level Tai Chi Activity Description"},
            {'title': 'Qi Gong', 'image': 'https://example.com/qi_gong_image.jpg', 'msg': "High Level Qi Gong Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Physical Exercises': [
            {'title': 'Hiking in nature', 'image': 'https://example.com/hiking_image.jpg', 'msg': "High Level Hiking in Nature Activity Description"},
            {'title': 'Group fitness classes', 'image': 'https://example.com/group_fitness_image.jpg', 'msg': "High Level Group Fitness Classes Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Gratitude Journaling': [
            {'title': 'Practice daily affirmations', 'image': 'https://example.com/daily_affirmations_image.jpg', 'msg': "High Level Practice Daily Affirmations Activity Description"},
            {'title': 'Mentor someone', 'image': 'https://example.com/mentor_someone_image.jpg', 'msg': "High Level Mentor Someone Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Creative Expression': [
            {'title': 'Painting', 'image': 'https://example.com/painting_image.jpg', 'msg': "High Level Painting Activity Description"},
            {'title': 'Photography', 'image': 'https://example.com/photography_image.jpg', 'msg': "High Level Photography Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
        'Social Activities': [
            {'title': 'Organize a community event', 'image': 'https://example.com/community_event_image.jpg', 'msg': "High Level Organize a Community Event Activity Description"},
            {'title': 'Lead a workshop', 'image': 'https://example.com/leading_workshop_image.jpg', 'msg': "High Level Lead a Workshop Activity Description"},
            # Add more activities and their corresponding image URLs and descriptions here
        ],
    }
}

# print(activity_data)

        
    # Determine the mental fitness rate category based on the predicted value
    if predicted_mental_fitness < 4.0:
        fitness_category = 'low'
    elif 4.0 <= predicted_mental_fitness < 7.0:
        fitness_category = 'moderate'
    else:
        fitness_category = 'high'

    # Retrieve recommended activities based on the mental fitness rate category and user's profile

    recommended_activities = []
    user_profile = ['Meditation', 'Mindfulness Exercises', 'Self-Care Routine', 'Relaxation Exercises', 
                    'Physical Exercises','Gratitude Journaling', "Creative Expression", 'Social Activities']
    for category, activities in activity_data[fitness_category].items():
        if category in user_profile:  # Check if the user is interested in this category
            recommended_activities.extend(activities)
    print(recommended_activities)
            
    return recommended_activities
    
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
    ip_df = pd.DataFrame(data=[[Country, year, schizophrenia, bipolar, eating_disorders, anxiety_disorders, drug_use_disorders, depressive_disorders, alcohol_use_disorders]],
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
    
  
    recommended_activities = recommendations(mental_fitness_rate)
    session['recommended_activities'] = recommended_activities

    
    
    # Prepare the response data
    response = {
        'prediction': prediction[0],
        'details': details
    }
    
    return render_template("result.html", data=response, round=round)



from flask import session

@app.route('/recommendations')
def recommendations_page():
    # Retrieve the recommended activities from the session
    recommended_activities = session.get('recommended_activities', [])
    
    # Clear the session variable to avoid displaying old recommendations on a new request
    session.pop('recommended_activities', None)
    
    return render_template("recommendations.html", recommended_activities=recommended_activities)


if __name__ == '__main__':
    app.run(debug=True)