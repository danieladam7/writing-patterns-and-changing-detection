# Stylometry Approach for Detecting Writing Style Patterns and Changes in Poetry text


### Objective
This project analyzes the stylistic changes in the poetry of Maya Angelou across different periods of her career using unsupervised learning and natural language processing techniques. The focus is on extracting and comparing stylistic topic modeling to understand the evolution of her writing style.

Each poetry is analyzed by the following features using NLTK tools, roBERTa and pretrained huggingface transformers and Gensim topic modelling.

- Basic Text Features: Document length, Mean Sentence Length, Mean Word Length, Readability
- Lexical Usage: Lexical richness, Function word frequencies, Content word frequencies
- Semantic Analysis: Semantic repetition
- Punctuation Usage
- Sentiment Analysis: Sentiment polarity (negative, neutral, and positive), Sentiment strength

Writing style patterns are detected with unsupervised learning using KNN and PCA reduction.



### RUN ME
To run this project, you will need to install the required Python libraries. You can install them using the following command:

pip install -r requirements.txt


### Project Structure
- `DataProcessor.py`: Contains functions for data preprocessing and feature extraction.
- `TextModeler.py`: Includes methods for text modeling and analysis.
- `Stylometry.py`: Tools for performing stylometric analysis.
- `Visualizer.py`: Provides visualization functions for the analysis results.
- `SemanticRepetitionDetector.py`: Detects semantic repetitions using BERT model


### Extracting the Project

After downloading the NLP_project.zip file, extract it to your desired location.

## Running the Analysis

1. Navigate to the `NLP_project` folder.
2. Ensure that the folders `initial state` and `final state` are present within the `Maya Angelou` folder.
3. Run the main script file using Python. Ensure you have all the required dependencies installed.

### Notes
- The script uses relative paths to access the corpus data. It expects the corpus folders to be in the `Maya Angelou` directory within the root project directory.
- If you encounter any path errors, please check that the folder structure matches the expected format and that the script is executed from the root project directory.


### IMPORTANT NOTE

The function def _semantic_repetition(self,text) in the class DataProcessor
will run some +5 minutes since it runs a BERT model which for each poem checks for semantic repetition.



Daniel Adam, I.D. 342475639
B.Sc. in Computer Science with specification in Data Science
Topics in Natural Language Processing
Ben Gurion University of the Negev

