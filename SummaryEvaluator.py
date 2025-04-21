from rouge import Rouge
from bert_score import score

class SummaryEvaluator:
  def __init__(self):
    self.rouge = Rouge()

  def calculate_rouge(self, generated_summary, reference_summary):
    """Calculate ROUGE scores"""
    scores = self.rouge.get_scores(generated_summary, reference_summary)
    return scores[0]

  def calculate_bertscore(self, generated_summary, reference_summary):
    """Calculate BERTScore"""
    try:
      P, R, F1 = score([generated_summary], [reference_summary], lang='en', verbose=False)
      return {
          'precision': P.item(),
          'recall': R.item(),
          'f1': F1.item()
      }
    except ImportError:
      print("Something wrong with BERTScore")
      return None

  def evaluate_summary(self, generated_summary, reference_summary):
    """Evaluate summary using multiple metrics"""
    results = {}
    results['rouge'] = self.calculate_rouge(generated_summary, reference_summary)
    bert = self.calculate_bertscore(generated_summary, reference_summary)
    if bert is not None:
      results['bertscore'] = bert
    return results
