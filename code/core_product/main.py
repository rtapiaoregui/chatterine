#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 09:26:05 2019

@author: rita

Flask app

"""

import pandas as pd
import re, random, os, time
from logic import handling
from funky import messenger, meta_enricher, pd_reader, pd_writer

from flask import Flask, render_template, request, session

app = Flask(__name__)
app.secret_key = b'_asgjDFGUsohjtsr3utew'

            
@app.route("/", methods=['POST', 'GET'])
def index():
    
    if request.method == 'POST':
        
        verbose = os.environ.get('VERBOSE', 'true').lower()=='true'
        msg_pth = '/Users/rita/Google Drive/chatterine/data/messages_{}.csv'.format(session['id'])
        new_obs0 = request.form['user_query']
        e, meta_data = handling(new_obs0, session['session'], session['id'], verbose)
        message = {'in': new_obs0, 'out': re.sub(r'<date_time_today>', 'the ' + str(time.strftime("%d-%m-%Y")), 
                                                 re.sub(r'\n', '<br>', e))}
        meta_enricher(meta_data)
        
        if session['session'] == 0:
            message = messenger([message], e)
        else:
            msg_df = pd_reader(msg_pth, index_col = 0)
            messages = msg_df.to_dict('records')
            messages.append(message)
            message = messenger(messages, e)
            
        pd_writer(pd.DataFrame(message), msg_pth)
                       
        session['messages'] = message
        
        # If the bot has determined the user wants to leave the conversation, 
        # the website should display a button, instead of the input bar. 
        if re.search(re.compile("Let's catch up sometime"), e):
            session['session'] = 666
        else:
            session['session'] += 1

    else:
        # Marks the beginning of a conversation.
        session['session'] = 0
        session['messages'] = []
        # I want a random conversation id to save the predictions and the scores 
        # belonging to a conversation in the files that are unique to that conversation.
        session['id'] = random.randint(0, 10**10)      

                        
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug = True, threaded = False)
    