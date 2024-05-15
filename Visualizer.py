import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import os


class Visualizer:
    def __init__(self):
        pass

    def plot_feature_distributions(self, feature_names, train_set, test_set): 
        plots_directory = r"Visualizations\Plots of feature distribution"      
        for i, feature_name in enumerate(feature_names):
            if feature_name == "Phrase Repetition":
                self._plot_phrase_repetition(train_set, test_set, plots_directory)
            else:
                plt.figure(figsize=(10, 6)) 
                
                # Plot the training set distribution
                sns.histplot(train_set[:, i], color='blue', label='Train Set', kde=True, alpha=0.5)
                # Plot the test set distribution
                sns.histplot(test_set[:, i], color='red', label='Test Set', kde=True, alpha=0.5)
                
                plt.title(f'Distribution of {feature_name}')
                plt.legend()
                plt.xlabel(feature_name)
                plt.ylabel('Frequency')
                
                if feature_name == "Sentiment Analysis":
                    # Create patches for the legend 
                    negative_patch = mpatches.Patch(color='red', label=' Negative Sentiment:  values in [-1,0)')
                    neutral_patch = mpatches.Patch(color='gray', label='Neutral Sentiment:    values around 0')
                    positive_patch = mpatches.Patch(color='green', label='Positive Sentiment: values in (0,1]')
                    
                    # Add the legend to the plot
                    plt.legend(handles=[negative_patch, neutral_patch, positive_patch])
                else:
                    plt.legend()
                                    
                self._save_plot(feature_name, '', plots_directory)
                plt.show()

            

    def plot_common_words(self, common_words, title):
        fig, ax = plt.subplots(figsize=(10, len(common_words) * 0.5 +1))
        bars = ax.barh(list(common_words.keys()), [1] * len(common_words), color = 'white')
        ax.set_yticks(list(range(len(common_words))))
        ax.set_yticklabels(list(common_words.keys()))
        ax.tick_params(axis='y', labelsize=10)   

        for bar, word_list in zip(bars, common_words.values()):
            words_text = ', '.join([f"{word}" for word, _ in word_list])
            ax.text(0.05, bar.get_y() + bar.get_height() / 2, f'   {words_text}',
                    va='center', ha='left', fontsize=10, color='black')

        ax.xaxis.set_visible(False)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title, fontsize=12)
        plots_directory = r"Visualizations\Plots of most common words"
        self._save_plot('', title, plots_directory)
        plt.show()


    def plot_common_POS(self, common_POS, title):
        POS_TAG_FULL_NAME = {
            'CC': 'Coordinating conjunction', 'CD': 'Cardinal number',
            'DT': 'Determiner', 'EX': 'Existential there', 'FW': 'Foreign word',
            'IN': 'Preposition',
            'JJ': 'Adjective', 'JJR': 'Adjective comparative',
            'JJS': 'Adjective superlative', 'LS': 'List item marker',
            'MD': 'Modal', 'NN': 'Noun singular',
            'NNS': 'Noun plural', 'NNP': 'Proper noun singular',
            'NNPS': 'Proper noun plural', 'PDT': 'Predeterminer',
            'POS': 'Possessive ending', 'PRP': 'Personal pronoun',
            'PRP$': 'Possessive pronoun', 'RB': 'Adverb',
            'RBR': 'Adverb comparative', 'RBS': 'Adverb superlative',
            'RP': 'Particle', 'SYM': 'Symbol', 'TO': 'to',
            'UH': 'Interjection', 'VB': 'Verb, base form',
            'VBD': 'Verb past tense', 'VBG': 'Verb gerund or present participle',
            'VBN': 'Verb past participle', 'VBP': 'Verb non-3rd person singular present',
            'VBZ': 'Verb 3rd person singular present', 'WDT': 'Wh-determiner',
            'WP': 'Wh-pronoun', 'WP$': 'Possessive wh-pronoun', 'WRB': 'Wh-adverb'
        }

        fig, ax = plt.subplots(figsize=(10, len(common_POS) * 0.5 +1))
        bars = ax.barh(list(common_POS.keys()), [1] * len(common_POS), color='white')
        for bar, pos_list in zip(bars, common_POS.values()):
            pos_text = ', '.join([POS_TAG_FULL_NAME.get(pos, 'Other') for pos, _ in pos_list])
            # Set the x position to be inside the bar
            ax.text(0.05, bar.get_y() + bar.get_height() / 2, pos_text, va='center', ha='left', fontsize=10)

            
        ax.xaxis.set_visible(False)            
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title, fontsize=12)
        plots_directory = r"Visualizations\Plots of most common POS"
        self._save_plot('', title, plots_directory) 
        plt.show()    


    def plot_common_structures(self, common_structures, title):
        fig, ax = plt.subplots(figsize=(10, len(common_structures) * 0.5 +1))

        bars = ax.barh(list(common_structures.keys()), [1] * len(common_structures), color='white')

        ax.xaxis.set_visible(False)
        ax.set_yticks(list(range(len(common_structures))))
        ax.set_yticklabels(list(common_structures.keys()))
        ax.tick_params(axis='y', labelsize=10)  

        for bar, structure in zip(bars, common_structures.values()):
            ax.text(0.05, bar.get_y() + bar.get_height() / 2, f'   {structure}',
                    va='center', ha='left', fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title, fontsize=12)
        plots_directory = r"Visualizations\Plots of most common structures"
        self._save_plot('', title, plots_directory) 
        plt.show()


    def plot_common_topics(self, common_topics, title):
        fig, ax = plt.subplots(figsize=(10, len(common_topics) * 0.5 +1))
        bars = ax.barh(list(common_topics.keys()), [1] * len(common_topics), color='white')
        ax.xaxis.set_visible(False)
        ax.set_yticks(list(range(len(common_topics))))
        ax.set_yticklabels(list(common_topics.keys()), fontsize=10)  

        for bar, topics in zip(bars, common_topics.values()):
            topic_text = ', '.join(topics[:2])  
            ax.text(0.05, bar.get_y() + bar.get_height() / 2, f'   {topic_text}',
                    va='center', ha='left', fontsize=10) 

        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.suptitle(title, fontsize=12)
        plots_directory = r"Visualizations/Plots of most common topics"
        self._save_plot('', title, plots_directory)
        plt.show()
  


    def plot_clusters(self, stylometry):
        reduced_train_data, reduced_test_data = stylometry.get_reduced_date()
        train_labels,test_labels = stylometry.get_labels()
        
        if reduced_train_data is None or reduced_test_data is None:
            raise ValueError("Train or test data not available for plotting.")
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(reduced_train_data[:, 0], reduced_train_data[:, 1], c=train_labels, cmap='viridis', marker='o', label='Train Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Train Data Clustering')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(reduced_test_data[:, 0], reduced_test_data[:, 1], c=test_labels, cmap='viridis', marker='o', label='Test Data')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Test Data Clustering')
        plt.legend()
        plots_directory = r"Visualizations\Plots of clustering"
        self._save_plot('','',plots_directory)
        plt.show()

        
    def _save_plot(self, feature_name, title, plots_directory):
        # Check if the directory exists, and if not, create it
        if not os.path.exists(plots_directory):
            os.makedirs(plots_directory)
        # Construct the full file path
        if feature_name != '':
            file_path = os.path.join(plots_directory, f'Distribution of {feature_name} in train and test corpus.png')
        elif title != '':
            file_path = os.path.join(plots_directory, f'{title}.png')
        else:
            file_path = os.path.join(plots_directory, f'Clustering of train and test corpus.png')
        plt.savefig(file_path,bbox_inches='tight')