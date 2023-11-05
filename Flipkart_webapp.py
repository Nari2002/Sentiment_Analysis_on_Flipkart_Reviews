import numpy as np
import pandas as pd
import streamlit as st
import joblib
import sklearn
model = joblib.load('flipkart_sentiment_model.pkl')

tfidi = joblib.load('tfidf_vectorizer_flipkart_data.pkl')

def analysis(input_text):
    input_data_features = tfidi.transform(input_text)
    data_features = pd.DataFrame(input_data_features.toarray())
 
    prediction = model.predict(data_features)
    print(prediction)
    if (prediction[0] == 0):
        return "Positive Sentiment :heart_eyes: :champagne: :tada:"
    elif (prediction[0] == 1):
        return "Negetive Sentiment :sneezing_face: :angry: :rage:"
    else:
        return "Neutral Sentiment :no_mouth: :slightly_smiling_face:"

def main():
    st.markdown("""
<style>
    /* Change the font size for all text within the Streamlit app */
    body {
        font-size: 40px;
    }
</style>
""", unsafe_allow_html=True)
    def set_bg_hack_url():
        '''
        A function to unpack an image from url and set as bg.
        Returns
        -------
        The background.
        '''
            
        st.markdown(
             f"""
             <style>
             .stApp {{
                 background: url("https://c.ndtvimg.com/2020-07/1k0ddgo_flipkart650_625x300_28_July_20.jpg?ver-20230922.06");
                 background-size: cover
             }}
             </style>
             """,
             unsafe_allow_html=True
         )
    set_bg_hack_url()
    st.title("Sentiment Analysis on Flipkart Reviews :shopping_trolley: ")
    input_text = st.text_input("Enter a Review on You Experience :thinking_face: :thinking_face:")
    
    dig =""
    if st.button("Analyse my sentiment 	:hugging_face:"):
        dig = analysis([input_text])
    st.success(dig)
        


if __name__ == '__main__':
    main()
