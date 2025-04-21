import streamlit as st


# Import existing classes
# We're assuming these are already defined in the notebook environment
from TextPreprocessor import TextPreprocessor
from ExtractiveSummarizer import ExtractiveSummarizer
from AbstractiveSummarizer import AbstractiveSummarizer


class TextSummarizer:
    def __init__(self, dataset_path="training_data.json"):
        with st.spinner("Initializing summarizers..."):
            self.preprocessor = TextPreprocessor()
            self.extractive_summarizer = ExtractiveSummarizer(self.preprocessor)
            # Initialize abstractive summarizer properly
            self.abstractive_summarizer = AbstractiveSummarizer()
            # Perform fine-tuning check during initialization
            self.abstractive_summarizer.check_and_fine_tune(dataset_path)
            self.dataset_path = dataset_path

    def summarize(self, text, method='both', ratio=0.3, max_length=700):
        """Generate summary using specified method"""
        result = {}

        if method in ["extractive", "both"]:
            with st.spinner("Generating extractive summary..."):
                result["extractive"] = self.extractive_summarizer.summarize(text, ratio=ratio)

        if method in ["abstractive", "both"]:
            with st.spinner("Generating abstractive summary..."):
                # Use the already initialized and fine-tuned summarizer
                result["abstractive"] = self.abstractive_summarizer.summarize(text, max_length=max_length)

        return result

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Text Summarizer",
        page_icon="📝",
        layout="wide"
    )

    st.title("Text Summarization Tool")
    st.write("Enter text below to generate extractive and abstractive summaries.")

    # Input text area
    text_input = st.text_area("Input Text", height=300,
                            placeholder="Paste your article or text here...")

    # Options for summarization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        summary_method = st.radio(
            "Summarization Method",
            ["both", "extractive", "abstractive"],
            index=0
        )

    with col2:
        if summary_method in ["extractive", "both"]:
            extract_ratio = st.slider(
                "Extractive Summary Ratio",
                min_value=0.1,
                max_value=0.5,
                value=0.3,
                step=0.05,
                help="Percentage of original text to keep in extractive summary"
            )
        else:
            extract_ratio = 0.3

    with col3:
        if summary_method in ["abstractive", "both"]:
            max_length = st.slider(
                "Max Abstractive Summary Length",
                min_value=100,
                max_value=1500,
                value=500,
                step=50
            )
        else:
            max_length = 500

    # Custom dataset path (for advanced users)
    # with st.expander("Advanced Settings"):
    #    dataset_path = st.text_input(
    #       "Training Dataset Path",
    #        value="training_data.json",
    #       help="Path to custom training data JSON or CSV file (format: list of objects with 'text' and 'summary' fields)"
    #    )
 
    # Create summarizer instance

    def get_summarizer(dataset_path="training_data.json"):
        return TextSummarizer(dataset_path="training_data.json")

    summarizer = get_summarizer(dataset_path="training_data.json")

    # Generate summary on button click
    if st.button("Generate Summary"):
        if not text_input.strip():
            st.error("Please enter some text to summarize.")
        else:
            with st.spinner("Generating summaries..."):
                summaries = summarizer.summarize(
                    
                    text_input,
                    method=summary_method,
                    ratio=extract_ratio,
                    max_length=max_length
                )

            # Display results
            if "extractive" in summaries:
                st.subheader("Extractive Summary")
                st.write(summaries["extractive"])
                st.markdown("---")

            if "abstractive" in summaries:
                st.subheader("Abstractive Summary")
                st.write(summaries["abstractive"])

            # Display statistics
            if "extractive" in summaries and "abstractive" in summaries:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Original Text Length", f"{len(text_input)} chars")
                with col2:
                    ext_reduction = 100 - (len(summaries["extractive"]) / len(text_input) * 100)
                    st.metric("Extractive Reduction", f"{ext_reduction:.1f}%")
                with col3:
                    abs_reduction = 100 - (len(summaries["abstractive"]) / len(text_input) * 100)
                    st.metric("Abstractive Reduction", f"{abs_reduction:.1f}%")

if __name__ == "__main__":
    main()