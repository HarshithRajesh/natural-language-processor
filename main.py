from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import wikipedia

nltk.download('wordnet')
nltk.download('averaged_perception_tagger')
nltk.download('punkt')

text = wikipedia.page('Vegetables').content

lemmatizer = WordNetLemmatizer()

def lemma_me(sent):
    sentence_tokens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tags(sentence_tokens)

    sentence_lemma = []
    for token,pos_tags in zip(sentence_tokens,pos_tags):
        if pos_tags[1][0].lower() in ['n','v','a','r']:
            lemma = lemmatizer.lemmatize(token,pos_tags[1][0].lower())
            sentence_lemma.append(lemma)
    
    return sentence_lemma

def process(text,question):
    sentence_tokens = nltk.sent_tokenize(text)
    sentence_tokens.append(question)

    tv = TfidfTransformer(tokenizer=lemma_me)
    tf = tv.fit_transform(sentence_tokens)
    values = cosine_similarity(tf[-1],tf)
    index = values.argsort()[0][-2]
    values_flat = values.flatten()
    values_flat.sort()
    coeff = values_flat[-2]
    if coeff > 0.3:
        return sentence_tokens[index]
    
    while True:
        question = input("Hi, what do you want to know?\n")
        output = process(text, question)
        if output:
            print(output)
        elif question=='quit':
            break
        else:
            print("I dont know.")
