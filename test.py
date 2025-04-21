import argparse
from TextPreprocessor import TextPreprocessor
from ExtractiveSummarizer import ExtractiveSummarizer
from AbstractiveSummarizer import AbstractiveSummarizer
from SummaryEvaluator import SummaryEvaluator

def test_preprocessor():
    print("\n=== Testing TextPreprocessor ===")
    preprocessor = TextPreprocessor()
    
    test_text = "This is a test sentence. This is another test sentence with numbers 123 and punctuation!"
    processed = preprocessor.preprocess_text(test_text)
    
    print(f"Original: {test_text}")
    print(f"Processed: {processed}")
    return preprocessor

def test_extractive_summarizer(preprocessor=None):
    print("\n=== Testing ExtractiveSummarizer ===")
    if preprocessor is None:
        preprocessor = TextPreprocessor()
    
    summarizer = ExtractiveSummarizer(preprocessor)
    
    test_text = """Machine learning (ML) is a field of study in artificial intelligence concerned with the development 
    and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks 
    without explicit instructions. Recent advances in machine learning have been driven by deep learning, 
    which uses neural networks with multiple layers. Deep learning has significantly improved computer vision, 
    speech recognition, natural language processing, and many other domains. The field of machine learning 
    continues to evolve with new techniques like transfer learning, reinforcement learning, and generative models 
    expanding the capabilities of artificial intelligence systems."""
    
    # Test different embedding methods
    for method in ['tf-idf', 'bow', 'word2vec']:
        try:
            summary = summarizer.summarize(test_text, embedding_method=method)
            print(f"\nEmbedding method: {method}")
            print(f"Summary: {summary}")
        except Exception as e:
            print(f"Error with {method}: {e}")
    
    return summarizer

def test_abstractive_summarizer():
    print("\n=== Testing AbstractiveSummarizer ===")
    summarizer = AbstractiveSummarizer()
    
    # Check if fine-tuning is needed
    summarizer.check_and_fine_tune("training_data.json")
    
    test_text = """Machine learning (ML) is a field of study in artificial intelligence concerned with the development 
    and study of statistical algorithms that can learn from data and generalise to unseen data, and thus perform tasks 
    without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning 
    have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning 
    approaches in performance."""
    
    summary = summarizer.summarize(test_text)
    print(f"Summary: {summary}")
    return summarizer

def test_evaluator(generated_summary, reference_summary):
    """Test the summary evaluator"""
    print("\n=== Testing SummaryEvaluator ===")
    evaluator = SummaryEvaluator()
    
    scores = evaluator.evaluate_summary(generated_summary, reference_summary)
    print("Evaluation Scores:")
    for metric, score in scores.items():
        print(f"{metric}: {score}")
    
    return evaluator

def main():
    parser = argparse.ArgumentParser(description='Test text summarization components')
    parser.add_argument('--test', choices=['all', 'preprocessor', 'extractive', 'abstractive', 'evaluator'], 
                       default='all', help='Which component to test')
    args = parser.parse_args()
    
    if args.test in ['all', 'preprocessor']:
        preprocessor = test_preprocessor()
    else:
        preprocessor = None
    
    if args.test in ['all', 'extractive']:
        extractive_summarizer = test_extractive_summarizer(preprocessor)
    
    if args.test in ['all', 'abstractive']:
        abstractive_summarizer = test_abstractive_summarizer()
    
    if args.test in ['all', 'evaluator']:
        generated_summary = "Machine learning is a field of artificial intelligence that learns from data."
        reference_summary = "Machine learning is an AI field focused on algorithms that learn from data to perform tasks."
        test_evaluator(generated_summary, reference_summary)
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()
