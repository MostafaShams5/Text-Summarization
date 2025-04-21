import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from TextPreprocessor import TextPreprocessor

class ExtractiveSummarizer:
  def __init__(self, preprocessor=None):
    """Initialize extractive summarizer"""
    self.preprocessor = preprocessor if preprocessor else TextPreprocessor()

  def _create_sentence_embeddings(self, sentences, method='tf-idf'):
    """Create sentence embeddings using tf-idf or word2vec or bow"""
    if method == 'tf-idf':
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(sentences)
        return embeddings

    elif method == 'bow':
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer = CountVectorizer()
        embeddings = vectorizer.fit_transform(sentences)
        return embeddings

    elif method == 'word2vec':
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            embeddings = model.encode(sentences)
            return embeddings
        except ImportError:
            # Fallback to tf-idf if sentence_transformers is not available
            print("SentenceTransformer not available, falling back to tf-idf")
            from sklearn.feature_extraction.text import TfidfVectorizer
            vectorizer = TfidfVectorizer()
            embeddings = vectorizer.fit_transform(sentences)
            return embeddings
    else:
        raise ValueError(f"Unsupported embedding method: {method}")

  def _build_similarity_matrix(self, sentences, embedding_method='tf-idf'):
    """Build similarity matrix between sentences using cosine similarity"""
    embeddings = self._create_sentence_embeddings(sentences, method=embedding_method)
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix

  def _pagerank(self, similarity_matrix, damping=0.85, iterations=30):
    """Run the PageRank algorithm on the similarity matrix"""
    n = len(similarity_matrix)
    scores = np.ones(n) / n

    for _ in range(iterations):
        new_scores = (1 - damping) * np.ones(n) / n
        for i in range(n):
            for j in range(n):
                if np.sum(similarity_matrix[j, :]) > 0:  # Avoid division by zero
                    new_scores[i] += damping * similarity_matrix[i, j] * scores[j] / np.sum(similarity_matrix[j, :])
        scores = new_scores

    return scores

  def summarize(self, text, ratio=0.2, min_length=2, embedding_method='tf-idf'):
    """Generate extractive summary"""
    sentences = self.preprocessor._split_sentences(text)
    if len(sentences) <= min_length:
        return text

    # Clean sentences for embedding but keep original for output
    cleaned_sentences = [self.preprocessor._clean_text(s) for s in sentences]
    similarity_matrix = self._build_similarity_matrix(cleaned_sentences, embedding_method=embedding_method)
    scores = self._pagerank(similarity_matrix)
    num_sentences = max(min_length, int(len(sentences) * ratio))

    ranked_sentences = [(i, score) for i, score in enumerate(scores)]
    ranked_sentences.sort(key=lambda x: x[1], reverse=True)

    top_sentence_indices = [idx for idx, _ in ranked_sentences[:num_sentences]]
    top_sentence_indices.sort()

    summary = ' '.join([sentences[i] for i in top_sentence_indices])
    return summary
