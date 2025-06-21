import re
import spacy
import random
import nltk
from nltk.corpus import stopwords

nlp = spacy.load('es_core_news_sm')
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

def short_corpus(input_path, output_path, percent=0.1, seed=42):
    random.seed(seed)

    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if random.random() <= percent:
                clean_line = clean_text(line)
                if clean_line:
                    outfile.write(clean_line + '\n')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", '', text)
    text = re.sub(r"[^a-zA-Z0-9áéíóúñüÁÉÍÓÚÑÜ\s]", '', text)
    text = re.sub(r"\d+", '', text)
    text = " ".join(text.split())
    return text

def preprocess_line(line):
    line = clean_text(line)
    doc = nlp(line)
    tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.lemma_) > 1]
    return " ".join(tokens)

if __name__ == "__main__":
    input_path = '../corpus/CoWeSe.txt'
    output_path = '../corpus/cowese_short.txt'
    short_corpus(input_path, output_path, percent=0.0003, seed=42)
