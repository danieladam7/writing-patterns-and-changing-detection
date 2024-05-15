from nltk import word_tokenize, pos_tag, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter
from gensim.models import LdaModel
from gensim import corpora
import os
import re
from string import punctuation



class TextModeler:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    
    def filter_POS(self, text):
        tokens = [word for word in word_tokenize(text) if word.isalpha() and word not in punctuation]
        pos_tags = pos_tag(tokens)
        # Filter for content words: nouns (NN), verbs (VB), adjectives (JJ), and adverbs (RB)
        # Exclude contractions like "n't" and non-ASCII characters
        filtered_tokens = [
            word.lower() for word, tag in pos_tags
            if tag.startswith(('NN', 'VB', 'JJ', 'RB')) and 
            word.lower() not in self.stop_words and 
            word.isalpha() and
            word.isascii() and
            not word.endswith("n't")
        ]
        return filtered_tokens
    
    def get_most_common_words_by_collection(self, directory):
        common_words_by_collection = {}
        for root, dirs, _ in os.walk(directory):
            for dir_name in dirs:
                collection_path = os.path.join(root, dir_name)
                all_words = []
                for filename in os.listdir(collection_path):
                    file_path = os.path.join(collection_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().lower()
                        words = word_tokenize(text)
                        all_words.extend([word for word in words if word.isalpha() and word not in self.stop_words])
                word_freq = Counter(all_words)
                most_common_words = word_freq.most_common(2)
                common_words_by_collection[dir_name] = most_common_words
        return common_words_by_collection

    
    def get_most_common_POS_by_collection(self, directory):
        common_pos_by_collection = {}
        for root, dirs, _ in os.walk(directory):
            for dir_name in dirs:
                collection_path = os.path.join(root, dir_name)
                all_tags = []
                for filename in os.listdir(collection_path):
                    file_path = os.path.join(collection_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().lower()
                        # Get POS tags for all words, not just filtered words
                        pos_tags = pos_tag(word_tokenize(text))
                        all_tags.extend([tag for _, tag in pos_tags])
                tag_freq = Counter(all_tags)
                most_common_tags = tag_freq.most_common(2)
                common_pos_by_collection[dir_name] = most_common_tags
        return common_pos_by_collection
    
    def get_most_common_structure_by_collection(self, directory):
        common_structure_by_collection = {}
        patterns = {
            'Question': re.compile(r'(\?)(?!\")'),
            'Exclamation': re.compile(r'(!)(?!\")'),
            'Starting Conjunction': re.compile(r'^(And|But|Or|So)[\s]')
        }
        for root, dirs, _ in os.walk(directory):
            for dir_name in dirs:
                collection_path = os.path.join(root, dir_name)
                all_structures = {key: 0 for key in patterns}
                for filename in os.listdir(collection_path):
                    file_path = os.path.join(collection_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        sentences = sent_tokenize(text)
                        for sentence in sentences:
                            for structure, pattern in patterns.items():
                                if pattern.search(sentence):
                                    all_structures[structure] += 1
                common_structure_by_collection[dir_name] = max(all_structures, key=all_structures.get)
        return common_structure_by_collection
    
    
    
    def get_most_common_topics_by_collection(self, directory, num_topics=2, num_words=20):
        common_topics_by_collection = {}
        for root, dirs, _ in os.walk(directory):
            for dir_name in dirs:
                collection_path = os.path.join(root, dir_name)
                all_filtered_text = []
                for filename in os.listdir(collection_path):
                    file_path = os.path.join(collection_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read().lower()
                        filtered_tokens = self.filter_POS(text)
                        all_filtered_text.extend(filtered_tokens)
                # Prepare the dictionary and corpus for LDA
                dictionary = corpora.Dictionary([all_filtered_text])
                corpus = [dictionary.doc2bow(text) for text in [all_filtered_text]]
                lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
                
                # Extract the topics
                topics = lda_model.show_topics(num_topics=num_topics, num_words=num_words, formatted=False)
                dominant_topics = sorted([(topic[0], topic[1]) for topic in topics], 
                                        key=lambda x: x[0], reverse=True)[:num_topics]
                common_topics_by_collection[dir_name] = [' '.join([word for word, _ in topic[1][:2]]) for topic in dominant_topics]
        return common_topics_by_collection
