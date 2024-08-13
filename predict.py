# Prediction client
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
 # Key class for azure
from msrest.authentication import ApiKeyCredentials
 # dotenv to load key
from dotenv import load_dotenv
 # Import os to read environment variables
import os

import gradio as gr

 # Load the key and endpoint values
load_dotenv()
    
 # Set the values into variables
key = os.getenv('KEY')
endpoint = os.getenv('ENDPOINT')
project_id = os.getenv('PROJECT_ID')
published_name = os.getenv('PUBLISHED_ITERATION_NAME')

 # Setup credentials for client
credentials = ApiKeyCredentials(in_headers={'Prediction-key':key})
    
 # Create client, which will be used to make predictions
client = CustomVisionPredictionClient(endpoint, credentials)
    
 # Open the test file
with open('testing-images/american-staffordshire-terrier-10.jpg', 'rb') as image:
     # Perform the prediction
     results = client.classify_image(project_id, published_name, image.read())
    
     # Because there could be multiple predictions, we loop through each one
     for prediction in results.predictions:
         # Display the name of the breed, and the probability percentage
         print(f'{prediction.tag_name}: {(prediction.probability):.2%}')


def predict_image(imagefile):
    with open(imagefile, 'rb') as image_data:
        results = client.classify_image(project_id, published_name, image_data.read())
        return {prediction.tag_name: (prediction.probability) for prediction in results.predictions}
    
demo = gr.Interface(
    fn = predict_image,
    inputs = gr.Image(type="filepath"),
    outputs = gr.Label(num_top_classes=5)
)

demo.launch()