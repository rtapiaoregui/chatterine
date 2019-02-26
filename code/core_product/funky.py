#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 19:31:13 2019

@author: rita

File with all the functions used in logic and main
(expect for the one to predict the labels of new observations).

"""

# Imports:
import pandas as pd
import os, re

import enchant
from google.cloud import storage
from io import StringIO


def guide():
    
    """
    Function to provide a dictionary with the paths to all the files 
    in the data folder that need to be loaded or saved.
    """
        
    data_pth = os.path.abspath('/Users/rita/Google Drive/chatterine/data')

    models_pth = os.path.join(os.path.dirname(data_pth), 'models')
    df_pth = os.path.join(data_pth, 'df_reduced.csv')
    onto_pth = os.path.join(data_pth, 'ontology.csv')
    df_new_obs = os.path.join(data_pth, 'obs_from_website.csv')
    df_4_scoring_pth = os.path.join(data_pth, 'df_4_scoring.csv')
    stopwords_pth = os.path.join(data_pth, 'stopwords.csv')
    named_entities_pth = os.path.join(data_pth, 'named_entities.csv')
    reference_pth = os.path.join(data_pth, 'reference.csv')
    vocabulary_pth = os.path.join(data_pth, 'vocabulary.csv')
    preds_pth = os.path.join(data_pth, 'preds.csv')
    
    labels = ['topic_level_0', 'topic_level_1', 'intent']
            
    paths = {
            'data_pth': data_pth, 'models_pth': models_pth, 
            'df_4_scoring_pth': df_4_scoring_pth, 'df_pth': df_pth, 'onto_pth': onto_pth, 
            'df_new_obs': df_new_obs, 'preds_pth': preds_pth, 'reference_pth': reference_pth, 
            'named_entities_pth': named_entities_pth, 'vocabulary_pth': vocabulary_pth, 
            'stopwords_pth': stopwords_pth, 'labels': labels
            }            
        
    return paths

        
def pd_reader(path, index_col=None, header='infer'):
    
    """
    Determines where a dataframe is supposed to be found and loads it.
    """
    
    if os.environ.get('SERVER_TYPE', '') == 'GCP':    
        filename = os.path.basename(path)
    
        gcs = storage.Client()
        bucket = gcs.get_bucket(os.environ.get('CLOUD_STORAGE_BUCKET'))
        blob = bucket.get_blob(filename)
        raw = blob.download_as_string()
    
        path_or_data = StringIO(str(raw,'utf-8')) 
    else:
        path_or_data = path
        
    if(index_col != None):
        df = pd.read_csv(path_or_data, index_col = index_col, header = header)
    else:
        df = pd.read_csv(path_or_data, header = header)
    return df


def pd_writer(df, path):
    
    """
    Determines where a dataframe is supposed to be stored and saves it.
    """

    if os.environ.get('SERVER_TYPE', '') == 'GCP': 
        filename = os.path.basename(path)
        gcs = storage.Client()
        bucket = gcs.get_bucket(os.environ.get('CLOUD_STORAGE_BUCKET'))
        blob = storage.blob.Blob(filename, bucket)
        blob.upload_from_string(df.to_csv())
    else:
        df.to_csv(path)
        
        
def delete_file(path):
    
    """
    Determines where a file is supposed to be found and, if it exists, 
    it deletes it.
    """

    if os.environ.get('SERVER_TYPE', '') == 'GCP': 
        filename = os.path.basename(path)
        gcs = storage.Client()
        bucket = gcs.get_bucket(os.environ.get('CLOUD_STORAGE_BUCKET'))
        blob = storage.blob.Blob(filename, bucket)
        if blob.exists():
            blob.delete()
    else:
        if os.path.exists(path):
            os.remove(path)

        
        
def meta_enricher(meta_data):
    
    """
    Joins all the user input the bot has received up until that point and 
    saves it in a separate file.
    """

    paths = guide()
    
    df = pd_reader(paths.get('df_new_obs'), index_col = 0)
    obs = pd.DataFrame(meta_data).T.reset_index(drop = True)
    df = df.append(obs, ignore_index=True, sort=False)
    pd_writer(df, paths.get('df_new_obs'))
    
            

def df_loader(df_pth):
    
    """
    Reads the dataframe with all the observations, which was saved to a file 
    in the Google Colab page 'annotation_tool' with just
    the explanatory and response variables, splits the target variable
    into the three different types of labels available, keeps only the observations
    belonging to labels with at least ten data point ascribed to them and returns 
    the dataframe used for training each of the models associated to one type of label.
    """
    
    df = pd_reader(df_pth, index_col = 0)
    assert df.shape[1] == 2
    props = df.target.value_counts()[(df.target.value_counts()>=10)].index.tolist()
    df1 = df[df.target.isin(props)][:]
    df1.dropna(inplace = True, how = 'any')
    df1.reset_index(drop = True, inplace = True)
    df1[['topic_level_0', 'topic_level_1', 'intent']] = df1.target.str.split(pat = '__', expand = True)
    df1.drop(columns = 'target', inplace = True)
    
    return df1


        
def spell_caster(data, named_ents, vocab, nlp, stopwords):
    
    """
    Checks for misspellings in a user query and replaces the misspelled words 
    when appropiate with the correctly spelled words that have 
    a higher likelihood of being the same words the user was referring to.
    """
    
    regexer = '(^({0})(?![\w])|(?<=[\W_])({0})(?![\w])|(?<=[\W_])({0})$|^({0})$)'
    spell_checker = enchant.Dict("en_US") 

    my_terms = set(vocab.vocabulary.str.lower())
    stopWords = set(stopwords)
    nes = set(sum([i.split('\b[\w\s\-]*\b') for i in named_ents.NE_value], []))
    my_terms.update(stopWords)
    my_terms.update(nes)

    # Parsing with Spacy
    processed_text = data.map(nlp).tolist()
    
    assert len(processed_text) == len(data)
    doc = []
    for idx in range(len(processed_text)):
        sent= []
        first_sent = data[idx]
        for a in processed_text[idx]:
            if a.is_alpha and not (spell_checker.check(a.orth_) or (a.orth_ in my_terms)):
                # I want to keep the candidates for replacing a word that appears 
                # misspelled in a user query that belong to my vocabulary, 
                # beacuse I consider they have a higher probability of being 
                # the more suitable ones.
                elem = [e for e in spell_checker.suggest(a.orth_) if e in my_terms]
                if elem:
                    b = elem[0]
                    replaced = re.sub(re.compile(regexer.format(a.orth_)), b, first_sent)
                    grammar1 = [i.pos_ for i in nlp(replaced) if i.orth_ == b]
                    if grammar1 and (a.pos_ != grammar1[0]) and len(elem) > 1:
                        replaced1 = re.sub(re.compile(regexer.format(a.orth_)), elem[1], first_sent)
                        grammar2 = [w.dep_ for w in nlp(replaced1) if w.orth_ == elem[1]]
                        if grammar2 and (a.pos_ != grammar2[0]) or (a.dep_ == grammar2[0]):
                            b = elem[1]
                    
                        sent.append(b)
                    else:
                        sent.append(a.orth_)

                else:
                    w = spell_checker.suggest(a.orth_)[0] if spell_checker.suggest(a.orth_) else a.orth_
                    sent.append(w)            
            else:
                sent.append(a.orth_)
        doc.append(' '.join(sent))

    data = pd.concat([data, pd.Series(doc)], axis = 1)
    data.columns = ['raw', 'doc']
    
    return data, named_ents, vocab



def beautifier(data, named_ents, vocab, nlp): 
    
    """
    Replaces the lemmatised words belonging to a named entity with the id of the same
    and returns the dataframe with the original and the preprocessed user query.
    """

    regexer = '(^({0})(?![\w])|(?<=[\W_])({0})(?![\w])|(?<=[\W_])({0})$|^({0})$)'
    doc = data.doc.map(nlp).tolist()
    
    # Lemmatising the user query
    lemmas = [' '.join([t.lemma_ if not re.search(r'[A-Z]', str(t.lemma_)) else t.orth_ for t in elem])
    for elem in doc]

    to_df = {'raw': data.doc.tolist(), 'lemmas': lemmas}    
    df_enriched = pd.DataFrame(to_df)
    
    keys = sorted(list(set(named_ents.NE_key)))
    
    ne_dict = {}
    for i in keys:
        to_compile = named_ents[named_ents.NE_key == i].NE_value                      
        ne_dict[i] = regexer.format('|'.join(to_compile.tolist()))
        
    replaced_ne = [df_enriched.lemmas.tolist()]
    for i in sorted(list(ne_dict.keys())):
        new_lemmas = []
        for idx in range(len(replaced_ne[-1])):
            new_lemmas.append(re.sub(re.compile(ne_dict.get(i), re.I), re.sub(r'\d?_', '', i), replaced_ne[-1][idx]))
        replaced_ne.append(new_lemmas)

    df_enriched = df_enriched.assign(clean_text=replaced_ne[-1])
    df_enriched.drop(columns = 'lemmas', inplace = True)
    
    return df_enriched



def facelifter(data, named_ents, vocab, nlp, stopwords):
    """ Wrapper function for the cleaning and preprocessing steps. """
    l, n, v = spell_caster(data, named_ents, vocab, nlp, stopwords)
    return beautifier(l, n, v, nlp)



def messenger(messages, e): 

    """
    Adds html font color tags to the response the bot gave to a previous query
    when it has been established as a valid response for the current query as well. 
    """
    rep_rex = re.compile("(?<=already\stold\syou\sthat\.\sTake\sa\scloser\slook\sat)[\w\s]*(last|\d+)")

    if rep_rex.search(e):
        listo = [d.get('out') for d in messages]
        if rep_rex.search(e).group().split(' ')[-1] == 'last':
            rogue_elem = listo[::-1][1]
            repl = '<font color="#3c69b2">' + rogue_elem + '</font>'

        else:
            pos = int(rep_rex.search(e).group())
            rogue_elem = listo[::-1][pos+1]
            repl = '<font color="#3c69b2">' + rogue_elem + '</font>'
        
        alt_mess = []
        for m in messages:
            for k, v in m.items():
                if k != 'out':
                    innies = {k:v} 
                else: 
                    if v == rogue_elem:
                        v = repl
                    else:
                        v = re.sub(re.compile('<font color="#3c69b2">'), '', re.sub(re.compile('</font>'), '', v))
                    innies.update({k:v})      
            alt_mess.append(innies)         
          
        messages = alt_mess
            
    return messages
    
    
    
    