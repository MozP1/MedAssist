import spacy
import re
import nltk
from nltk.corpus import stopwords


nltk.download('stopwords')


stop_words = set(stopwords.words('english'))

nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
   
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words]

    return ' '.join(tokens)


if __name__ == "__main__":
    sample_text = "Patient presents with chest pain, shortness of breath, and fatigue."
    print(preprocess_text(sample_text))
   
