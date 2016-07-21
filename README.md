# POS-Tagging

Problem:
In corpus linguistics, part-of-speech tagging (POS tagging or POST), also called grammatical tagging or word-category disambiguation, is the process of marking up a word in a text (corpus) as corresponding to a particular part of speech, based on both its definition, as well as its context.

Implementation:
- Implemented a deep network to tackle the problem of POS Tagging
- Deep network trained with features including word contexts, preceding and succeeding words and orthographic text features to classify words

Future Lines of Work:
- Model to use look-up lists for punctuation marks and filter numbers before tagging words in dataset
- Learning Word Embeddings as features

Imports: (keep these up to date to run code)
- Theano
- Keras wrapper for theano
- SKlearn
- Numpy

Instructions:
- Execute POS.py file from terminal or an IDE
- Train and text data (included in .gz form) should be in the same directory as the main python script
