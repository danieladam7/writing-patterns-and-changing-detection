from DataProcessor import DataProcessor
from TextModeler import TextModeler
from Stylometry import Stylometry
from Visualizer import Visualizer




### extracting data from corpus ###
train_corpus = r'Maya Angelou\initial state'
test_corpus = r'Maya Angelou\final state'



data_processor = DataProcessor()
text_modeler = TextModeler()
stylometry = Stylometry()
visualizer = Visualizer()



# extraxt feature from train and test corpus
train_set = data_processor.create_feature_matrix(train_corpus)
test_set = data_processor.create_feature_matrix(test_corpus)


# Get the most common words, POS, text structures and topics by collection from train and test corpus
common_words_train = text_modeler.get_most_common_words_by_collection(train_corpus)
common_words_test = text_modeler.get_most_common_words_by_collection(test_corpus)

common_POS_train = text_modeler.get_most_common_POS_by_collection(train_corpus)
common_POS_test = text_modeler.get_most_common_POS_by_collection(test_corpus)

common_structures_train = text_modeler.get_most_common_structure_by_collection(train_corpus)
common_structures_test = text_modeler.get_most_common_structure_by_collection(test_corpus)

common_topics_train = text_modeler.get_most_common_topics_by_collection(train_corpus)
common_topics_test = text_modeler.get_most_common_topics_by_collection(test_corpus)




# Visualize distributions of features
feature_names = data_processor._get_feature_names()
visualizer.plot_feature_distributions(feature_names,train_set,test_set) 

### apply stylometry on extracted feature matrixes of train and test data ###

stylometry.fit(train_set)
test_predictions = stylometry.predict(test_set)
# Visualize the clustering
visualizer.plot_clusters(stylometry)

# Visualize most common words
visualizer.plot_common_words(common_words_train, 'Most common non-stop words in train corpus')
visualizer.plot_common_words(common_words_test, 'Most common non-stop words in test corpus')

# Visualize most common POS
visualizer.plot_common_POS(common_POS_train, "Most common POS's in train corpus")
visualizer.plot_common_POS(common_POS_test, "Most common POS's in test corpus")

# Visualize most common text structures
visualizer.plot_common_structures(common_structures_train, "Most common structure in train corpus")
visualizer.plot_common_structures(common_structures_test, "Most common structure in test corpus")

# Visualize most common topics
visualizer.plot_common_topics(common_topics_train, "Most common topics in train corpus")
visualizer.plot_common_topics(common_topics_test, "Most common topics in test corpus")


