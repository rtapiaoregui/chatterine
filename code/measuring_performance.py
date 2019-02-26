#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:46:46 2019

@author: rita

Code to evaluate the chatbot's performance

"""

import pandas as pd
import re, os

from core_product.funky import df_loader, guide
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt


def evaluator(raw):
    
    """ 
    Adds a new column to the dataframe with the information on whether 
    an intent was correctly predicted or not.
    """
    
    y_hat = raw[raw.notna().values].filter(regex=("pred"))
    label = y_hat.index.values[0].split('_pred')[0]
    if (y_hat.values[0] == raw[label]):
        score = 'hit_' + label
    else:
        score = 'missed_' + label
    return score



def prep_df_4_scoring(paths):
    
    """
    Loading the file containing the conversations held with chatterine
    as well as the file with the data that has already been annotated, including 
    that of previous interactions, which is on which the bot's performance 
    can be measured, and merging both data sets with an inner join.
    """
   
    df = df_loader(paths.get('df_pth'))
    df.set_index('utterance', inplace = True)
    
    df_preds = pd.read_csv(paths.get('df_new_obs'), index_col=0)
    df_preds.set_index('utterance', inplace = True)
        
    df0 = df_preds.merge(df, how = 'inner', left_index=True, right_index=True)
    df0.reset_index(inplace = True)
    df0 = df0.assign(score=df0.apply(evaluator, axis=1))

    df0.to_csv(paths.get('df_4_scoring_pth'))
    
    return df0



def conf_mat_preps(cm):
    
    """ Plots a confusion matrix. """
    
    plt.figure()
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap = plt.get_cmap('Blues'))
    plt.tight_layout()
    plt.title("Confusion matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.colorbar()
    plt.show()



def scoring():
    
    """ Shows the model's perfomance. """
    
    paths = guide()
    
    if not os.path.exists(paths.get('df_4_scoring_pth')):
        df = prep_df_4_scoring(paths)
    else:
        df = pd.read_csv(paths.get('df_4_scoring_pth'), index_col = 0)
    
    print(df.score.value_counts())
    print()
    df[df.score.str.contains('hit_intent')].shape[0]/df.shape[0]
    df = df.applymap(lambda x: re.sub('nan', '__fallback__', str(x))) 
    
    cm = {}
    for i in sorted(list(set(df.intent))):
        d = df[df['intent'] == i][:]
        for p in sorted(list(set(df.intent_pred))):
            a = d[d['intent_pred'] == p]
            cm.setdefault(i, []).append(len(set(a.index.values)))                
        
    cm_df = pd.DataFrame(cm)
    cm_df.set_index(pd.Index(sorted(list(set(df.intent_pred)))), inplace = True)
    
    conf_mat_preps(cm_df.values)
    print()
    print(classification_report(df.intent, df.intent_pred))


if __name__ == '__main__':
    
    scoring()