#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 12:01:24 2019

@author: rita

Code to train the ML models.

"""
import pandas as pd
import numpy as np
import re, os
from collections import Counter
import en_core_web_sm

from core_product.funky import facelifter, df_loader, guide

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def NE_loader(named_entities_pth, nlp):
    
    """
    Reads the csv-file directly downloaded from the google sheet 'named_entities', 
    to be found in the spreadsheet 'data', lemmatizes the named entity values and 
    replaces the blank spaces between words with a regular expression to match up to 
    two words. 
    """
        
    named_ents = pd.read_csv(re.sub(r'named_entities', 'named_entities0', named_entities_pth), header = None)
    named_ents.columns = ['NE_key', 'NE_value0']
    named_ents.drop_duplicates(inplace = True)
    named_ents.drop_duplicates(subset = 'NE_value0', keep = False, inplace = True)
    named_ents = named_ents.assign(len_val=named_ents.NE_value0.map(len))
    named_ents.set_index(['NE_key', 'NE_value0'], inplace = True)
    named_ents.sort_values(by=['len_val'], inplace = True)
    named_ents.reset_index(inplace = True)
    vs = named_ents.NE_value0.map(nlp)
    vs1 = [[str(t.lemma_) if not re.search(r'[A-Z]', str(t.lemma_)) else str(t.orth_) for t in elem] 
    for elem in vs]
    vs2 = [r'\b([\w\-_]*\s){0,3}\b'.join(el) for el in vs1]
    named_ents = named_ents.assign(NE_value=pd.Series(vs2))
    named_ents.reset_index(inplace = True, drop = True)
    named_ents.to_csv(named_entities_pth)



def reference_compiler(reference_pth, df0, named_ents, vocab, stopwords, nlp):
    
    """
    Builds a dataframe with the least frequently occurring lemmatised nouns, 
    adjectives and verbs of each emission over all emissions 
    and saves it into a file. 
    """
       
    regexer = '(^({0})(?![\w])|(?<=[\W_])({0})(?![\w])|(?<=[\W_])({0})$|^({0})$)'
    
    ner_set = set(sum(list(map(lambda x: x.split(' '), named_ents.NE_value0.tolist())), []))
    ner_set.update(set(stopwords))
    ner_set.update(set(vocab))
    
    parsed = nlp(' '.join(df0.emission.tolist()).lower())
    elem = [(t.lemma_, t.tag_) if not re.search(r'[A-Z]', str(t.lemma_)) else (t.orth_, t.pos_) for t in parsed]
    list_to_counter = [e for e, p in elem if (len(e) > 2) and re.match(r'(N|V|JJ)', str(p))]
    
    # I just want the terms that are not part of my glossary of technical terms.
    counter = Counter([lc for lc in list_to_counter if not lc in ner_set])
    new_c = [k for k, v in counter.items() if v < 4]
    rex = re.compile(regexer.format('|'.join(new_c)), re.I)
    
    vs = df0.emission.map(lambda x: nlp(x.lower()))
    vs1 = [' '.join([t.lemma_ if not re.search(r'[A-Z]', str(t.lemma_)) else t.orth_ for t in elem])
    for elem in vs]

    special_wxi = {}
    for i in range(df0.shape[0]):
        for t in re.findall(rex, vs1[i]):
            special_wxi.setdefault(df0.intent.iloc[i], []).append(t[0])
            
    previous_e_w = {k: regexer.format('|'.join(v)) for k, v in special_wxi.items()}
    
    reference = pd.DataFrame(list(previous_e_w.items()), columns = ['prev_pred', 'regex'])
    reference.to_csv(reference_pth)
        
            


def models_trainer(X, y, named_ents, vocab, label):
    
    """
    Vectorises the input and trains a GridSearchCV with a logistic regression 
    as estimator and a changing value for C, the regularisation parameter.    
    """

    if re.search(r'0', label):
        params = {'clf__C': np.logspace(-0.1, 1.6, num=30)}
    
    elif re.search(r'1', label):
        params = {'clf__C': np.logspace(2, 3, num=30)}
    
    else:
        params = {'clf__C': np.logspace(1.5, 2.5, num=30)}
        
    chars_vec = TfidfVectorizer(ngram_range = (4, 10), analyzer = 'char_wb', min_df = 3)
    lemma_vec = TfidfVectorizer(ngram_range = (1, 5), analyzer = 'word', min_df = 2)

    clf = LogisticRegression(verbose = 2)
    
    if re.search(r'_proba', label):
        clf.predict = clf.predict_proba
    
    pipe = Pipeline([
            ('features', ColumnTransformer([
                    ('chars', chars_vec, 0),
                    ('lemmas', lemma_vec, 1),
                    ])),
            ('clf', clf)])
    
    model = GridSearchCV(pipe, param_grid=params, cv=10, scoring = 'f1_weighted')
    
    # Training
    model.fit(X, y)
        
    return model.best_estimator_



def personal_trainer():

    """ 
    Trains each of the three models that are used 
    to predict the different type of labels and saves them in files.
    """    

    paths = guide() 
    nlp = en_core_web_sm.load()
            
    df = df_loader(paths.get('df_pth'))
    df0 = pd.read_csv(paths.get('onto_pth')) 

    if not os.path.exists(paths.get('named_entities_pth')):
         NE_loader(paths.get('named_entities_pth'), nlp)
         
    named_ents = pd.read_csv(paths.get('named_entities_pth'), index_col = 0)
    vocab = pd.read_csv(paths.get('vocabulary_pth')) 
    stopwords = pd.read_csv(paths.get('stopwords_pth')).iloc[:,1]
    
    if not os.path.exists(paths.get('reference_pth')):
        reference_compiler(paths.get('reference_pth'), df0, named_ents, vocab, stopwords, nlp)

    # Preprocessing
    X = facelifter(df.utterance, named_ents, vocab, nlp, stopwords)
    X.rename(columns = {'raw': '0_raw', 'clean_text': '1_clean_text', 
                        'pos_tags': '2_pos_tags', 'syn_deps': '3_syn_deps'}, inplace = True)
    
    X = X.reindex(sorted(X.columns), axis=1)
                     
    for l in paths.get('labels')[::-1]:          
        # Separating the target variable  
        y = df.pop(l)

        model_pth = os.path.join(paths.get('models_pth'), l+'.joblib')

        # Training the model
        model = models_trainer(X, y, named_ents, vocab, l)
        # Saving the trained model
        with open(model_pth, 'wb') as fh:
            joblib.dump(model, fh)
                
  
if __name__ == '__main__':
   
    personal_trainer()
    
    