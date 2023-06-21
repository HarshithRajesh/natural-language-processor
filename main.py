from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('wordnet')
x='was'
lemmatizer = WordNetLemmatizer()
lemma = lemmatizer.lemmatize(x,'v')
print(lemma)
