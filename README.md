# Text Summarization Tool

A comprehensive text summarization tool that provides both extractive and abstractive summarization capabilities using NLP techniques.

## Features

- **Extractive Summarization**: Identifies and extracts key sentences from a document using TextRank algorithm
- **Abstractive Summarization**: Generates concise summaries with new phrasing using a fine-tuned BART model
- **Interactive UI**: Streamlit-based interface for easy text input and summary generation
- **Evaluation Tools**: ROUGE and BERTScore metrics for summary quality assessment

## Screenshots

### Streamlit Interface
![Streamlit Interface](screenshots/streamlit_interface.png)

### Summary Output Example
![Summary Example](screenshots/summary_example.png)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MostafaShams5/Text-Summarization.git
cd Text-Summarization
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download training data or prepare your own:
```bash
# The training_data.json file should be in the root directory
# Format: [{"text": "original text", "summary": "reference summary"}, ...]
```

## Usage

### Web Interface

Run the Streamlit app:
```bash
streamlit run app.py
```

### From Python

```python
from TextPreprocessor import TextPreprocessor
from ExtractiveSummarizer import ExtractiveSummarizer
from AbstractiveSummarizer import AbstractiveSummarizer

# Initialize summarizers
preprocessor = TextPreprocessor()
extractive_summarizer = ExtractiveSummarizer(preprocessor)
abstractive_summarizer = AbstractiveSummarizer()

# Generate summaries
text = "Your long text here..."
extractive_summary = extractive_summarizer.summarize(text)
abstractive_summary = abstractive_summarizer.summarize(text)
```

## Project Structure

- `app.py`: Streamlit web application
- `TextPreprocessor.py`: Text cleaning and preprocessing utilities
- `ExtractiveSummarizer.py`: Extractive summarization implementation
- `AbstractiveSummarizer.py`: Abstractive summarization implementation
- `SummaryEvaluator.py`: Evaluation metrics for summaries
- `test.py`: Test script for summarizers
- `training_data.json`: Training data for fine-tuning the abstractive model

## Model Fine-tuning

The abstractive summarizer uses a fine-tuned version of facebook/bart-large-cnn. Due to size limitations, the fine-tuned model is not included in the repository. When you first run the application, it will check for a fine-tuned model and train one using the provided training data if necessary.

## Future Improvements

- Support for more languages
- Implement more summarization algorithms
- Enhanced UI with summary comparisons

## License

[MIT License](LICENSE)
