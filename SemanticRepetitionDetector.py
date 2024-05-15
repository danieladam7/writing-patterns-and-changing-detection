from transformers import BertTokenizer, BertModel
import torch
from nltk.tokenize import sent_tokenize
import numpy as np

class SemanticRepetitionDetector:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def _get_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(1)

    def count_repetitions(self, text, threshold=0.9):
        sentences = sent_tokenize(text)
        embeddings = [self._get_embeddings(sentence).detach().numpy() for sentence in sentences]
        repetition_count = 0
        
        # Compare each sentence to every other sentence
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j].T) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                if sim > threshold:
                    repetition_count += 1
        return repetition_count