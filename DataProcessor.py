import os
import string

import nltk
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk import pos_tag, ngrams
from nltk.corpus import cmudict
from textstat import textstat

from collections import Counter

from transformers import pipeline
from transformers import BertTokenizer
from SemanticRepetitionDetector import SemanticRepetitionDetector







# Ensure required NLTK downloads
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')



class DataProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        try:
            self.cmu_dict = {word: min([len([y for y in pron if y[-1].isdigit()]) for pron in prons])
                             for word, prons in cmudict.entries()}
            print("CMU Dictionary successfully loaded and processed.")
        except Exception as e:
            print(f"Failed to load or process CMU Dictionary: {e}")
            self.cmu_dict = {}

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

        self.feature_names = [
                            'Document Length',
                            'Mean Sentence Length',
                            'Mean Word Length',
                            'Readability',
                            'Lexical Richness',  
                            'Semantic Repetition',                          
                            'Function Words Frequency',
                            'Content Words Frequency',
                            'Punctuation Usage', 
                            'Sentiment Indicator',
                            'Sentiment Strength'
                        ]  

        
    def _get_feature_names(self):
        return self.feature_names
    
    # functions for features
    ### Features for Phraseology ###
    def _document_length(self, text):
        return len(word_tokenize(text))
    
    def _mean_sentence_length(self, text):
        sentences = sent_tokenize(text)
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        return np.mean(sentence_lengths) if sentence_lengths else 0

    def _mean_word_length(self, text):
        tokens = [word for word in word_tokenize(text) if word.isalpha()]  # Ignore punctuation
        lengths = [len(word) for word in tokens]
        return np.mean(lengths) if lengths else 0

    ### Features for Lexical Usage ###
    def _readability(self, text):
        return textstat.flesch_reading_ease(text)
      
    def _lexical_richness(self, text):
        tokens = word_tokenize(text)
        types = len(set(tokens))
        tokens_total = len(tokens)
        return types / tokens_total if tokens_total > 0 else 0
    
               
    def _semantic_repetition(self,text):
        semantic_repeteition = SemanticRepetitionDetector()
        return semantic_repeteition.count_repetitions(text)

    
    def _function_words_frequency(self, text):
        tokens = word_tokenize(text.lower())
        function_words = [word for word in tokens if word in self.stop_words]
        return len(function_words) / len(tokens) if tokens else 0
    
    def _content_words_frequency(self, text):
        # Tokenize and apply POS tagging
        tokens = word_tokenize(text)
        words_and_pos = pos_tag(tokens)
        content_pos_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
        content_words = [word for word, pos in words_and_pos if pos in content_pos_tags and word.lower() not in self.stop_words]
        content_word_count = len(content_words)
        total_words = len(tokens)
        return content_word_count / total_words if total_words > 0 else 0   
    
        
    
    ### Features for Punctuation Usage ###
    def _punctuation_usage(self, text):
        punctuations = Counter(char for char in text if char in string.punctuation)
        return sum(punctuations.values())


    ### Features for Sentiment Analysis ###     

    def _sentiment_indicator(self, text):
        segments = self._split_into_segments(text)
        # Process in batches
        results = self.sentiment_pipeline(segments)

        sentiment_score = sum(1 if res['label'] == 'POSITIVE' else -1 if res['label'] == 'NEGATIVE' else 0 for res in results)

        if sentiment_score > 0:
            return 1  # Positive sentiment
        elif sentiment_score < 0:
            return -1  # Negative sentiment
        else:
            return 0  # Neutral sentiment
        
    def _sentiment_strength(self, text):
        segments = self._split_into_segments(text)
        if not segments:
            return 0  # Return early if no segments

        # Process all segments at once using batch processing
        results = self.sentiment_pipeline(segments)

        # Initialize sentiment counts
        sentiment_counts = {1: 0, -1: 0, 0: 0}

        # Update counts based on results from the batch processing
        for res in results:
            label = 1 if res['label'] == 'POSITIVE' else -1 if res['label'] == 'NEGATIVE' else 0
            sentiment_counts[label] += 1

        # Return the count of the most predominant sentiment
        predominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        return sentiment_counts[predominant_sentiment]

    def _split_into_segments(self, text, max_length=510):
        # Use the BERT tokenizer to tokenize the text and manage the max token length
        tokens = self.tokenizer.tokenize(text)
        segments = []
        current_segment = []
        
        for token in tokens:
            if len(current_segment) + 1 > max_length:
                segments.append(self.tokenizer.convert_tokens_to_string(current_segment))
                current_segment = []
            current_segment.append(token)
        
        if current_segment:
            segments.append(self.tokenizer.convert_tokens_to_string(current_segment))
        
        return segments

    def _extract_features(self, text):
        features = [
            # Phraseology
            self._document_length(text),
            self._mean_sentence_length(text),
            self._mean_word_length(text),
            #Lexical
            self._readability(text),
            self._lexical_richness(text),   
            self._semantic_repetition(text),      
            self._function_words_frequency(text),
            self._content_words_frequency(text),
            # Punctuation
            self._punctuation_usage(text),
            # Sentiment
            self._sentiment_indicator(text),
            self._sentiment_strength(text),
        ]
        return features

    def create_feature_matrix(self, directory):
        features_matrix = []
        for root, _, files in os.walk(directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    features = self._extract_features(text)
                    features_matrix.append(features)
        return np.array(features_matrix)

