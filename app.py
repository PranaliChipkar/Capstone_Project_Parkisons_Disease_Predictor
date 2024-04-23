import streamlit as st
from tensorflow import keras
load_model = keras.models.load_model

import numpy as np

import pandas as pd

st.set_page_config(
    page_title = 'Parkinson\'s Disease Prediction',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)



# Load your trained generator model
try:
    generator = load_model('generator_model.h5')
    classifier = load_model('discriminator_model.h5')
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# Define the app
def main():
    st.markdown("""
        <style>
        /* All text elements */
              
        /* Background color for the entire app */
        .stApp {
            background-color: #12151c; 
            
        }         
             
        /* Background color for the sidebar */
        .st-emotion-cache-14rvwma {
            background-color:#1b222c;  /* Darker background for sidebar */
        }
        h1,h2,h3,p {
                color:#ffffff;
        }        
        .st-emotion-cache-18ni7ap {
            background-color: #12151c; 
        }
                
        .st-emotion-cache-hc3laj {
            background-color: #ff4b4b;
        }
       }
        
        </style>
         """, unsafe_allow_html=True)

    
 
    st.sidebar.header('Parkinson’s Disease Data Generator using GAN')
    num_samples = st.sidebar.number_input('Number of samples to generate', min_value=1, max_value=100, value=5)
    generate_button = st.sidebar.button('Generate Data', help='Click here to generate synthetic data')

    # Container for generated data
    with st.container():
        # Generate data button
        if generate_button:
            with st.spinner('Generating synthetic data using GAN ...'):
                noise = np.random.normal(0, 1, (num_samples, 100))
                generated_samples = generator.predict(noise)

                # Define feature names
                feature_names = [
                    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                    'spread1', 'spread2', 'D2', 'PPE'
                ]

                generated_df = pd.DataFrame(generated_samples, columns=feature_names)

                # Style the DataFrame
                st.write("Generated Data Samples")
                st.dataframe(generated_df.style.format(subset=pd.IndexSlice[:, :], formatter="{:.2f}"))

                # Predict using the classifier
                predictions = classifier.predict(generated_samples)
                binary_predictions = (predictions > 0.5).astype(int)

                # Display predictions
                st.write("Predictions on GAN-generated Data")
                predictions_df = pd.DataFrame(binary_predictions, columns=['Prediction'])
                st.dataframe(predictions_df.style.format(formatter="{:.0f}"))



# Import pages
from Tabs import home, data, predict, visualise



# Dictionary for pages
Tabs = {
    "Home": home,
    "Data Info": data,
    "Prediction": predict,
    "Visualisation": visualise,
   
    
}

# Create a sidebar
# Add title to sidear
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio(" Choose a Page", list(Tabs.keys()))

from web_functions import load_data
# Loading the dataset.
df, X, y = load_data()

# Call the app funciton of selected page to run
if page in ["Prediction", "Visualisation"]:
    Tabs[page].app(df, X, y)
elif (page == "Data Info"):
    Tabs[page].app(df)
else:
    Tabs[page].app()
# Run the app
if __name__ == '__main__':
    main()

