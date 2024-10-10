# Local GenAI Search: Generative Search Engine for Local Files

## Description

Local GenAI Search is an AI-powered document search and question-answering system that allows users to explore and extract information from their local documents using natural language queries. This application combines semantic search capabilities with generative AI to provide accurate and context-aware answers based on the content of your local files.

## Features

- **Document Indexing**: Supports indexing of PDF, DOCX, PPTX, and TXT files.
- **Semantic Search**: Utilizes FAISS and sentence transformers for efficient semantic search.
- **AI-powered Question Answering**: Generates answers to user queries using the Ollama AI model.
- **User-friendly Interface**: Built with Streamlit for an intuitive and interactive user experience.
- **Document Reference**: Provides citations and links to source documents for generated answers.
- **File Download**: Allows users to download referenced documents directly from the interface.

## Requirements

- Python 3.7+
- See `requirements.txt` for a full list of dependencies

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/imanoop7/Generative-Search-Engine-For-Local-Files
   cd Generative-Search-Engine-For-Local-Files
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have Ollama installed and the 'tinyllama' model available.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run local_genai_search.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Enter the path to your documents folder in the text input field.

4. Click the "Index Documents" button to process and index your files (this step is required only once or when you add new documents).

5. Once indexing is complete, you can start asking questions about your documents in the "Ask a Question" section.

6. The AI will provide answers based on the content of your indexed documents, along with references to the source materials.

## How It Works

1. **Document Indexing**: The system reads and chunks your documents, then creates embeddings using a sentence transformer model. These embeddings are stored in a FAISS index for efficient similarity search.

2. **Semantic Search**: When you ask a question, the system converts it into an embedding and finds the most similar document chunks in the FAISS index.

3. **Answer Generation**: The relevant document chunks are used as context for the Ollama AI model, which generates a comprehensive answer to your question.

4. **Result Presentation**: The answer is displayed along with references to the source documents, which can be expanded to view the full context or downloaded for further review.

## Contributing

Contributions to improve Local GenAI Search are welcome! Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

 MIT License
