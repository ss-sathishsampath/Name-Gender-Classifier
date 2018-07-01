# -*- coding: utf-8 -*-
"""
Name Gender Classifier using NLTK

@author: Sathish Sampath(ss.sathishsampath@gmail.com)
Developed as part of  Microsoft's NLP MOOC(https://www.edx.org/course/natural-language-processing-nlp)

"""

# code to build a classifier to classify names as male or female
# demonstrates the basics of feature extraction and model building
import random
import nltk
import numpy


names = [(name, 'male') for name in nltk.corpus.names.words("male.txt")]
names += [(name, 'female') for name in nltk.corpus.names.words("female.txt")]

def extract_gender_features(name):
    name = name.lower()
    features = {}
    features["suffix"] = name[-1:]
    features["suffix2"] = name[-2:] if len(name) > 1 else name[0]
    features["suffix3"] = name[-3:] if len(name) > 2 else name[0]
    features["suffix4"] = name[-4:] if len(name) > 3 else name[0]
    features["suffix5"] = name[-5:] if len(name) > 4 else name[0]
    features["suffix6"] = name[-6:] if len(name) > 5 else name[0]
    features["prefix"] = name[:1]
    features["prefix2"] = name[:2] if len(name) > 1 else name[0]
    features["prefix3"] = name[:3] if len(name) > 2 else name[0]
    features["prefix4"] = name[:4] if len(name) > 3 else name[0]
    features["prefix5"] = name[:5] if len(name) > 4 else name[0]
    features["wordLen"] = len(name)
    
#    for letter in "abcdefghijklmnopqrstuvwyxz":
#        features[letter + "-count"] = name.count(letter)
   
    return features



data = [(extract_gender_features(name), gender) for (name,gender) in names]
random.shuffle(data)


dataCount = len(data)
trainCount = int(.8*dataCount)

trainData = data[:trainCount]
testData = data[trainCount:]
bayes = nltk.NaiveBayesClassifier.train(trainData)

def classify(name):
    label = bayes.classify(extract_gender_features(name))
    print("name=", name, "classifed as=", label)

print("trainData accuracy=", nltk.classify.accuracy(bayes, trainData))
print("testData accuracy=", nltk.classify.accuracy(bayes, testData))

# Display best features
bayes.show_most_informative_features(25)

# print gender classifier errors so we can design new features to identify the cases
errors = []

for (name,label) in names:
    if bayes.classify(extract_gender_features(name)) != label:
        errors.append({"name": name, "label": label})

print("Errors")
print(errors)
