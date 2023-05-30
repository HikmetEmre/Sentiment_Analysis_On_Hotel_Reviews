### New App For Hotel Guests Reviews ###
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
import streamlit as st

#### Page Config ###
st.set_page_config(
    page_title="Sentiment Analysis on Hotel Guests Reviews",
    page_icon="https://thehubbackend.com/media/64814-shutterstock_1073953772.jpg",
    menu_items={
        "Get help": "mailto:hikmetemreguler@gmail.com",
        "About": "For More Information\n" + "https://github.com/HikmetEmre/Sentiment_Analysis_On_Hotel_Reviews"
    }
)

### Title of Project ###
st.title("**:red[A NLP Project For Creating Model To Analysis Hotel Guests Reviews.]** ")

### Markdown ###
st.markdown("**Introducing a fascinating :red[NLP project] that delves into the world of hotel reviews!**.")

### Adding Image ###
st.image("https://www.travelmediagroup.com/wp-content/uploads/2022/06/bigstock-Stress-Level-Mood-Scale-Man-447658594-2880x1568.jpg")

st.markdown("**This project aims to understand the sentiments expressed in customer reviews and identify the most important words that contribute to positive or negative experiences.**")
st.markdown("**In this project, a dataset containing :red[1 million hotel reviews] will be used to identify the words that have the strongest positive or negative impact on the overall sentiment of a review.**")
st.markdown("**The ultimate goal is to build a predictive model that can classify new reviews as :blue[positive] or :red[negative] based on the words used in the text. The insights gained from this project can be useful for hotel owners and managers to better understand customer feedback and improve their services accordingly.**")
st.markdown("*Alright, Let's Look What's Happening In The Kitchen!*")

st.image("https://getthematic.com/assets/img/sentiment-analysis/aspect-based-sentiment-analysis.png")

#### Header and definition of columns ###
st.header("**META DATA**")

st.markdown("- **Sentiment**: Sentiment refers to the emotional or subjective attitude expressed in text.")
st.markdown("- **Rating**: A review rating refers to a numerical or categorical score assigned to a product, service, or experience by a reviewer.")
st.markdown("- **Text**:A written or textual expression that provides a detailed description, opinion,service or experience.")
st.markdown("- **Polarity**:Typically ranging from -1 to +1, where negative values indicate negative sentiment, positive values indicate positive sentiment.")
st.markdown("- **New_Sentiment**: A new ranging values based on our analysis.")



### Example DF ON STREAMLIT PAGE ###
df=pd.read_csv("sampled_data.csv")


### Example TABLE ###
st.table(df.sample(5, random_state=17))

st.image("https://raw.githubusercontent.com/HikmetEmre/Sentiment_Analysis_On_Hotel_Reviews/main/top_neg_words.png")

#---------------------------------------------------------------------------------------------------------------------

### Sidebar Markdown ###
st.sidebar.markdown("**INPUT** The **:red[Text]** Of Guest to see the Result Of **:blue[Sentiment Analysis.]**")

### Define Sidebar Input's ###
Review = st.sidebar.text_input("**:blue[A text as feedback about hotel guests experiences.]**")


#---------------------------------------------------------------------------------------------------------------------

### Recall Model ###
from joblib import load

nlp_model = load('mnb_model.pkl')

input_df = pd.DataFrame({"Text":[Review]})
    
### For fit StandartScaler ###
df=pd.read_csv("sampled_data.csv")

# Define X and y
X = df.text

y = df.sentiment


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the data
from sklearn.feature_extraction.text import CountVectorizer
cv1 = CountVectorizer(stop_words='english') ### Getting Rid Of Stop Words ###

### Transforming Our Data with CountVectorizer ###
X_train_cv1 = cv1.fit_transform(X_train)
X_test_cv1  = cv1.transform(X_test)



### Scale the new input data###


input_df_scaled = cv1.transform(input_df)

pred = nlp_model.predict(input_df_scaled)






#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

### Result Screen ###
if st.sidebar.button("Submit"):

    ### Info message ###
    st.info("You can find the result below.")

    ### Inquiry Time Info ###
    from datetime import date, datetime

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    ### For showing results create a df ###
    results_df = pd.DataFrame({
    'Date': [today],
    'Time': [time],
    'Text': [Review],
    'Sentiment': [pred]
    })

   


    st.table(pred)

    if pred == 'Positive':
        st.image("https://raw.githubusercontent.com/HikmetEmre/Sentiment_Analysis_On_Hotel_Reviews/main/positive%20image.png")

    elif pred == 'Neutral':
        st.image("https://raw.githubusercontent.com/HikmetEmre/Sentiment_Analysis_On_Hotel_Reviews/main/neutral%20image.png")

    elif pred == 'Negative':
        st.image("https://raw.githubusercontent.com/HikmetEmre/Sentiment_Analysis_On_Hotel_Reviews/main/negative%20image.png")    
else:
    st.markdown("Please click the *Submit Button*!")
