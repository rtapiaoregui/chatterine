#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 15:40:32 2019

@author: rita

Chatterine's reasoning for choosing to respond one way or another.

"""

import pandas as pd
import numpy as np
import re, resource, sys, time

from funky import facelifter, guide, pd_reader, pd_writer
from predict import predict
import en_core_web_sm



def handling(new_obs0, s, s_id, verbose = False):
    
    """
    Function to choose the emission chatterine should provide to the user 
    in response to his query. 
    
    Parameters
    ----------
        new_obs0 (str): the query
        s (int): the amount of queries the user has entered
        s_id (int): the unique id of the conversation taking place
        verbose (bool): indication as to whether the time and memory 
            consumed by each operation should be printed out.
        
    Returns
    -------
        bot_emission (str): chatterine's response
        complete_score (dict): the dictionary with the user input, the prediction 
            corresponding to the emission provided by the bot as well as 
            the model's confidence on its prediction.
        
    """
          
    start = time.time()
    paths = guide()        
    df0 = pd_reader(paths.get('onto_pth'))
    df0 = df0.iloc[1:]
    named_ents = pd_reader(paths.get('named_entities_pth'), index_col = 0)
    vocab = pd_reader(paths.get('vocabulary_pth')) 
    stopwords = pd_reader(paths.get('stopwords_pth')).iloc[:,1]
    reference =  pd_reader(paths.get('reference_pth'), index_col = 0)
    pred_pth = re.sub(r'\.', str(s_id), paths.get('preds_pth'))

    if not bool(int(s)):
        predictions = []
    else:
        pred_df = pd_reader(pred_pth, index_col = 0)
        predictions = pred_df.iloc[:, 0].tolist()
        
    if len(predictions) >= 13:
        del predictions[0]
       
    if verbose:
        end = time.time()
        print('\nIt has taken {0} second(s) to load the data.'.format(round(end-start, 2)), file=sys.stderr)
        print('Memory usage is {0} Mb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), file=sys.stderr)

    if re.search(r'[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,3}', new_obs0):
        bot_emission = "Thanks! Your email has been saved correctly."
        predictions.append("email")
        pd_writer(pd.DataFrame(predictions), pred_pth)
        score = [{new_obs0 : 
            {'utterance' : new_obs0,
             'intent_score': None, 
             'intent_pred': "email"}}]
        complete_score = {k: v for d in score for k, v in d.items()}
        return bot_emission, complete_score

    start = time.time()
    nlp = en_core_web_sm.load(disable=['ner', 'textcat'])
    if verbose:
        end = time.time()
        print('\nIt has taken {0} second(s) to load the nlp model.'.format(round(end-start, 2)), file=sys.stderr)
        print('Memory usage is {0} Mb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), file=sys.stderr)
        start = time.time()
        
    new_obs1 = facelifter(pd.Series([new_obs0.lower()]), named_ents, vocab, nlp, stopwords)
    if verbose:
        end = time.time()
        print('\nIt has taken {0} second(s) to clean and preprocess the text.'.format(round(end-start, 2)), file=sys.stderr)
        print('Memory usage is {0} Mb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), file=sys.stderr)

            
    if (re.search(r'exit|(^see\sy\w{1,3})|^(i\s([^\s]+\s){0,3})?(go|leave|later|say\s(good)?bye)(\s\w+){0,2}\W*$', 
                  new_obs1.clean_text[0]) and not 
        re.search(r'\brita\b', new_obs1.clean_text[0])):
        bot_emission = "It's been a pleasure to talk to you. Let's catch up sometime."
        predictions.append("exit_conversation")
        pd_writer(pd.DataFrame(predictions), pred_pth)
        
        score = [{new_obs0 : 
            {'utterance' : new_obs0,
             'intent_score': None, 
             'intent_pred': "exit_conversation"}}]
        complete_score = {k: v for d in score for k, v in d.items()}
        return bot_emission, complete_score
        
    # I want to loop over the 3 different types of available labels in the following order:
    # 'intent', 'topic_level_1', and finally, 'topic_level_0'
    
    # If the prediction of the label belonging to the type presently under consideration
    # meets the necessary requirements to be considered reliable enough, 
    # the emission attached to it will be returned.
    
    # Otherwise, the next type of labels will be predicted.
    scores = []
    for l in paths.get('labels')[::-1]:

        start = time.time()
        prediction, pred_prob = predict(new_obs1, paths, l)
        if verbose:
            end = time.time()
            print('\nIt has taken {0} second(s) to predict one observation.'.format(round(end-start, 2)))
            print('Memory usage is {0} Mb'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024), file=sys.stderr)

        if (l == paths.get('labels')[-1]):
            pred_opt1 = prediction
            int_pr_prob = pred_prob
            int_model_cls = np.asarray(sorted(list(set(df0[l]))))
            emission = 'emission'
        else:
            emission = 'fallback_' + l

        response0 = list(set(df0[df0[l] == prediction][emission]))[0]
        score = {'utterance' : new_obs0,
                 l+'_score': pred_prob.max()*10**2, 
                 l+'_pred': prediction}

        scores.append({new_obs0: score})
        complete_score = {k: v for d in scores for k, v in d.items()}
    
        # The second-best prediction:
        mask = pred_prob.argsort()[-1][-2]
        predict_opt2 = np.asarray(sorted(list(set(df0[l]))))[mask]
        probas_preds_opt2 = pred_prob[0][mask]
        response1 = list(set(df0[df0[l] == predict_opt2][emission]))[0]
        if len(predictions) <= 4:
            param_m_pr = 0.80
            param_best_intent = 0.70
            param_2best_pr = 0.50
        elif len(predictions) > 4 and len(predictions) <= 8:
            param_m_pr = 0.70
            param_best_intent = 0.55
            param_2best_pr = 0.35
        else:
            param_m_pr = 0.60
            param_best_intent = 0.40
            param_2best_pr = 0.25
      
        if ((pred_prob.max() <= param_best_intent) or 
            (predictions and (prediction in predictions))):
            
            if (predictions and (l == paths.get('labels')[-1]) and 
                (pred_prob.max() <= param_best_intent) and (predictions[-1] in set(reference.prev_pred)) and 
                re.search(re.compile(reference[reference.prev_pred == predictions[-1]].regex.iloc[0]), new_obs0.lower())):
                bot_emission = df0[df0.intent == 'elaborate_request'].emission.iloc[0]
                predictions.append("elaborate_request")
                pd_writer(pd.DataFrame(predictions), pred_pth)
                break
            
            elif (predictions and (l == paths.get('labels')[-1]) and (prediction in predictions) and 
                  (pred_prob.max() > param_m_pr)):
                if predictions[::-1].index(prediction) == 0:
                    bot_emission = "I thought I'd already told you that. Take a closer look at the last message I sent you."
                    predictions.append("repeated_info")
                    pd_writer(pd.DataFrame(predictions), pred_pth)
                    break
                else:
                    bot_emission = "I thought I'd already told you that. Take a closer look at what I sent you {} messages ago.".format(str(predictions[::-1].index(prediction)+1))
                    predictions.append("repeated_info")
                    pd_writer(pd.DataFrame(predictions), pred_pth)
                    break

            else:
                # The list of intents belonging to the predicted topic. 
                viable_preds = list(set(df0[df0[l] == prediction][paths.get('labels')[-1]]))
                mask = pred_prob.argsort()[-1][-2]
                pred_opt2 = int_model_cls[mask]
                probas_opt2 = int_pr_prob[0][mask]
                response01 = list(set(df0[df0[paths.get('labels')[-1]] == pred_opt1].emission))[0]
                response02 = list(set(df0[df0[paths.get('labels')[-1]] == pred_opt2].emission))[0]

                if pred_opt1 in viable_preds and not (pred_opt1 in predictions):
                    bot_emission = response01
                    predictions.append(pred_opt1)
                    pd_writer(pd.DataFrame(predictions), pred_pth)
                    break
                elif ((pred_opt2[1] in viable_preds) and (probas_opt2 >= param_2best_pr-0.1) and not
                      (pred_opt2 in predictions)):
                    bot_emission = response02
                    predictions.append(pred_opt2)
                    pd_writer(pd.DataFrame(predictions), pred_pth)
                    break
                else:
                    if l == paths.get('labels')[0]:
                        if ((prediction in predictions) and not 
                            (predict_opt2 in predictions)):
                            bot_emission = response1
                            predictions.append(predict_opt2)
                            pd_writer(pd.DataFrame(predictions), pred_pth)
                            break
                        else:
                            bot_emission = response0
                            predictions.append(prediction)
                            pd_writer(pd.DataFrame(predictions), pred_pth)
                            break

                                        
        elif (predictions and (prediction in predictions and not 
              ((predict_opt2 in predictions) or (probas_preds_opt2[0] < param_2best_pr)))):
            bot_emission = response1
            predictions.append(predict_opt2)
            pd_writer(pd.DataFrame(predictions), pred_pth)
            break
          
        else:
            bot_emission = response0
            predictions.append(prediction)
            pd_writer(pd.DataFrame(predictions), pred_pth)
            break                
                
    return bot_emission, complete_score

    

if __name__ == '__main__':
    
    bot_es = []
    metas = []
    s = 0
    for i in ["enter here the sentences you would like to test logic on", 
              "example sentence number 2", 
              "example sentence number 3",
 # ...
              "and so on"]:
        bot_emission, meta_data = handling(i, s, 'fghjklö')
        s += 1
        bot_es.append(bot_emission)
        metas.append(meta_data)
        
        
        
        
        
        
        
        