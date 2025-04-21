import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
  def __init__(self, language='english'):
    """
    Initialize the text preprocessor

    Args:
        language: Language for stopwords and stemming
    """
    self.stop_words = set(stopwords.words(language))
    self.lemmatizer = WordNetLemmatizer()

  def _clean_text(self, text):
    """Clean the text using regular expressions (regex)"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

  def _remove_stopwords(self, text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word not in self.stop_words]
    return ' '.join(filtered_words)

  def _lemmatize_text(self, text):
    words = word_tokenize(text)
    lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

  def _split_sentences(self, text):
    return sent_tokenize(text)

  def preprocess_text(self, text, clean=True, remove_stops=True, lemmatize=True, split=True):
    """Apply full preprocessing pipeline"""
    if split:
      texts = self._split_sentences(text)
    else:
      texts = [text]

    processed_sentences = []
    for text in texts:
      if clean:
        text = self._clean_text(text)
      if remove_stops:
        text = self._remove_stopwords(text)
      if lemmatize:
        text = self._lemmatize_text(text)
      processed_sentences.append(text)
    return processed_sentences
