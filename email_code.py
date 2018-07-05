import sys
from nltk.tokenize.stanford import StanfordTokenizer
import re
import csv
import os
import operator
from numpy import dot
from numpy.linalg import norm
import pdb
import numpy as np
import string
f = open('emails_dataset/emails.dataset','r')

import sent2vec
model = sent2vec.Sent2vecModel()
model.load_model('model_bi.bin')

def tokenize(tknzr, sentence, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentence: a string to be tokenized
        - to_lower: lowercasing or not
    """
    sentence = sentence.strip()
    sentence = ' '.join([format_token(x) for x in tknzr.tokenize(sentence)])
    if to_lower:
        sentence = sentence.lower()
    sentence = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))','<url>',sentence) #replace urls by <url>
    sentence = re.sub('(\@ [^\s]+)','<user>',sentence) #replace @user268 by <user>
    filter(lambda word: ' ' not in word, sentence)
    return sentence

def format_token(token):
    """"""
    if token == '-LRB-':
        token = '('
    elif token == '-RRB-':
        token = ')'
    elif token == '-RSB-':
        token = ']'
    elif token == '-LSB-':
        token = '['
    elif token == '-LCB-':
        token = '{'
    elif token == '-RCB-':
        token = '}'
    return token

def tokenize_sentences(tknzr, sentences, to_lower=True):
    """Arguments:
        - tknzr: a tokenizer implementing the NLTK tokenizer interface
        - sentences: a list of sentences
        - to_lower: lowercasing or not
    """
    return [tokenize(tknzr, s, to_lower) for s in sentences]

SNLP_TAGGER_JAR = "/usr0/home/anushap/DirectedStudy/SEM2/sent2vec/src/stanford-postagger-2018-02-27/stanford-postagger.jar"

# In[3]:


lis = f.readlines()


# In[4]:


emails = []
emails_len = []
gold = []
for each in lis:
    item = each.split("\t")
    gold.append(item[1]) 
    item_email =  ' '.join(e for e in item[2:])
    item_email = item_email.replace('</br>',' ').replace(':',' ')#.split()
    #print(item_email)
    emails.append(item_email)
    emails_len.append(len(item_email))

max_words_email = max(emails_len)
total_email = len(emails)
embedding_size = 300
all_mail_vec = []

tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
s = ' <delimiter> '.join(emails)
tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
emails = tokenized_sentences_SNLP[0].split(' <delimiter> ')


embs_email = model.embed_sentences(emails)  

#each_mail_vec = np.zeros((max_words_email, embedding_size))
#print(vector.shape)
f.close()

f = open('emails_dataset/emailExplanations_Dec23.sorted.txt','r')

concepts = {'REMINDER':[], 'HUMOR' : [], 'EVENT': [], 'EMPLOYEE': [], 'MEETING' : [], 'POLICY' : [], 'CONTACT' : []}
lis = f.readlines()
for each in lis:
    item = each.split("\t")
    t1 = item[2].split("_")
    item_email =  ' '.join(e for e in item[3:])
    item_email = item_email.replace('</br>',' ').replace(':',' ')#.split()
    #print(item_email)
    concepts[item[1]].append((int(t1[0]),item_email))
    #print(concepts)

tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')

desc_key = {}
for k,v in concepts.items():
    desc_sent = []
    desc_weight = []
    for each in v:
        desc_weight.append(each[0])
        desc_sent.append(each[1])
    s = ' <delimiter> '.join(desc_sent)
    tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
    desc_sent = tokenized_sentences_SNLP[0].split(' <delimiter> ')
    desc_key[k] = (desc_weight, desc_sent)
        
tot_cat = 7


'''
total_desc = len(desc)
max_words_desc = max([len(x[1]) for x in desc])


# In[30]:
tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')
s = ' <delimiter> '.join(desc)
tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
desc = tokenized_sentences_SNLP[0].split(' <delimiter> ')

# In[6]:
'''

#embs_desc = model.embed_sentences(desc)
print(len(embs_email))
pred = []
for item in embs_email:
     cos = {}
     prod_key = {}
	#mini = 0
     for k,v in concepts.items():
        embs_desc = model.embed_sentences(desc_key[k][1])
        each_cos = []
        for ix, itemd in enumerate(embs_desc):
          val = dot(item, itemd)/(norm(item)*norm(itemd))
          #val = cosine_similarity(item,itemd)
          #if val>mini:
          #   mini_index = ix
          #   mini = val
          each_cos.append(val)
        cos[k] = each_cos

        #prod = [a*b for a,b in zip(cos[k],desc_key[k][0])]
        #prod = [x/len(desc_key[k][0]) for x in prod]
        prod = [a*b for a,b in zip(cos[k],[1]*len(desc_key[k][0]))]
        prod_key[k] = np.sum(prod)/len(desc_key[k][0])
        #(sum((desc_key[k][0])))
        #/len(desc_key[k][0])
        #(sum((desc_key[k][0])))
        #prod_key[k] = np.sum(prod)/(len(desc_key[k][0])*sum((desc_key[k][0])))
     pred.append(max(dict((key,value) for key, value in prod_key.items() if key == 'POLICY' or key == 'CONTACT').items(), key=operator.itemgetter(1))[0])

accuracy = len([pred[i] for i in range(0, len(pred)) if pred[i] == gold[i] and gold[i] == 'CONTACT' or gold[i] == 'POLICY']) / len([pred[i] for i in range(0, len(pred)) if gold[i] == 'CONTACT' or gold[i] == 'POLICY'])
print(accuracy)

rows = zip(emails, gold, pred)
with open('bi_.csv', 'w') as f:
	writer = csv.writer(f)
	for row in rows:
		writer.writerow(row)
