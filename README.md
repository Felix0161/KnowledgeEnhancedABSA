# LCR-Rot-hop-ont++
The code for Injecting Knowledge Using a Domain Sentiment Ontology in Neural Models for Aspect-Based Sentiment Analysis

This is an extension of the HAABSA++ model (https://github.com/mtrusca/HAABSA_PLUS_PLUS). The ontology is used as an external knowledge base out of which knowledge is injected into the test-data sentences.

 
 # Data
The data for the semEval2015 and semEval2016 tasks, as well as the ontology are included in the data/externalData directory or can be downloaded via the following links:
- SemEval2015: http://alt.qcri.org/semeval2015/task12/index.php?id=data-and-tools
- SemEval2016: http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools. 
- Ontology: https://github.com/KSchouten/Heracles/tree/master/src/main/resources/externalData

# Setup Environment
1. Open a new virtual environment in Anaconda (using Python version 3.7)
2. Clone the GitHub page https://github.com/Felix0161/KnowledgeEnhancedABSA
3. Install the packages in the requirements.txt file (in your command terminal: pip install -r requirements.txt)

# Preferences
- The config file lets you choose the number of hops, inclusion of softpositioning and inclusion of visibility matrices.
- You can choose to not include sub-classes and/or super-classes by setting include_subclasses and/or include_superclasses to 'False' in the KnowledgeBranch file.

# Software
- Run the generate_data.py file, to obtain the train and test data (already added to the programGeneratedData directory in Github)
- Run the loadEmbeddings.py file(in the directory embeddings)
  - NOTE: the data sets are in general too large for creating all the embeddings at once. In the config file you can set a start and an end to be able to run it in batches. We found that splitting the semEval2016 data set in four (e.g., 0-850,850-1700,1700-2100, and 1700-2530 sentences) and the semEval2015 in three (e.g., 0-800, 800-1300, 1300-1880 sentences) suffices.
- Run the main_hyper file for hyperparameter optimization
- Run the main file to obtain the accuracy
- Run the main_cross file for k-fold cross validation

# Related Work
This code uses source code from the following papers:
- Liu, W., Zhou, P., Zhao, Z., Wang, Z., Ju, Q., Deng, H., Wang, P.: K-BERT: Enabling language representation with knowledge graph. In: 34th AAAI Conference on Artificial Intelligence (AAAI 2020). vol. 34, pp. 2901–2908. AAAI (2020)
- Trusca, M.M., Wassenberg, D., Frasincar, F., Dekker, R.: A hybrid approach for aspect-based sentiment analysis using deep contextual word embeddings and hierarchical attention. In: 20th International Conference on Web Engineering (ICWE 2020). LNCS, vol. 12128, pp. 365–380. Springer (2020)
- Wallaart, O., Frasincar, F.: A hybrid approach for aspect-based sentiment analysis using a lexicalized domain ontology and attentional neural models. In: 16th Extended Semantic Web Conference (ESWC 2019). LNCS, vol. 11503, pp. 363–378. Springer (2019)

Parts of the code are based on https://github.com/DanaeGielisse/LCR-Rot-hop-ont-plus-plus

