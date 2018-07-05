import sys
from nltk.tokenize.stanford import StanfordTokenizer
import numpy as np
from itertools import chain
import re
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

def calculate_grad(weight, lamb, X, y):

    temp = np.dot(X,weight)
    exp = np.exp(temp)
    summ = np.sum(exp,axis=1).reshape(exp.shape[0],1)

    val = np.divide(exp,summ)
    diff = y - val
    diff = np.dot(X.T,diff)
    grad = diff - lamb*weight

    return grad

def evaluate(weight, X):

    temp = np.dot(X,weight)
    exp = np.exp(temp)
    summ = np.sum(exp,axis=1).reshape(exp.shape[0],1)

    val = np.divide(exp,summ)

    hard = np.argmax(val, axis=1) + 1

    return hard

def calculate_loss(weight, lamb, X, y):

    temp = np.dot(X,weight)
    exp = np.exp(temp)
    summ = np.sum(exp,axis=1).reshape(exp.shape[0],1)

    val = np.divide(exp,summ)

    y = y-1
    val = val[np.arange(len(val)), y.tolist()]

    log = np.log(val)
    summ = np.sum(log)

    loss = summ - (lamb/2)*np.sum(np.power( np.linalg.norm(weight,axis=0),2))

    return loss

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
embedding_size = 700
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

concepts_dic = {'REMINDER':1, 'HUMOR' : 2, 'EVENT': 3, 'EMPLOYEE': 4, 'MEETING' : 5, 'POLICY' : 6, 'CONTACT' : 7}

gold_new = []
for each in gold:
	gold_new.append(concepts_dic[each])

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

embs_desc = []

for k,v in concepts.items():
        embs_desc.append(model.embed_sentences(desc_key[k][1]))

embs_desc_new = list(chain.from_iterable(embs_desc))


embs_desc_new = np.vstack(embs_desc_new)
embs_email = np.vstack(embs_email)

#weight = np.zeros((embedding_size, embedding_size))
weight = np.zeros((embs_desc_new.shape[0],7))
#final = np.dot(np.dot(embs_email,weight),embs_desc_new.T)
final_all = np.dot(embs_email,embs_desc_new.T)

#train, test, label_train, label_test = train_test_split(final_all, gold_new, test_size = 0.1)
#training, test = x[:80,:], x[80:,:]order = np.arange(train.shape[0])

order = np.arange(final_all.shape[0])
np.random.shuffle(order)

gold_new = np.array(gold_new)[order]
final_all = final_all[order]

train, test = final_all[:824,:], final_all[824:,:]
label_train, label_test = np.array(gold_new)[:824], np.array(gold_new)[824:]

max_epoch = 10000
learning_rate = 0.01
decay_rate = 0.0001
lamb = 0.1
#3631.3203125
#7230.875

for epoch in range(max_epoch):

        lr = learning_rate/(1+decay_rate*epoch)
        order = np.arange(train.shape[0])
        np.random.shuffle(order)

        n = train.shape[0] 
       
        batch_label = np.array(label_train)[order]
        batch_data = train[order] 

        bb = np.zeros((len(batch_label), 7))
        bb[np.arange(len(batch_label)), batch_label-1] = 1

        grad = calculate_grad(weight, lamb, batch_data, bb)
        weight = weight + (lr * grad/n)

        hard = evaluate(weight, test)
  
        accuracy_test = np.sum(label_test == hard)/n

        loss_test = calculate_loss(weight, lamb, test, label_test)
   
        hard = evaluate(weight, batch_data)

        accuracy_train = np.sum(batch_label == hard)/n

        loss_train = calculate_loss(weight, lamb, batch_data, batch_label)

        print("Epoch : ", epoch, "AccTest : ",accuracy_test, "AccTrain : ",accuracy_train, "LossTest : ",loss_test, "LossTrain, : ",loss_train)

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
        pdb.set_trace()
        each_cos = []
        for ix, itemd in enumerate(embs_desc):
          val = dot(item, itemd)/(norm(item)*norm(itemd))
          #val = cosine_similarity(item,itemd)
          #if val>mini:
          #   mini_index = ix
          #   mini = val
          each_cos.append(val)
        cos[k] = each_cos

        prod = [a*b for a,b in zip(cos[k],desc_key[k][0])]
        #prod = [x/len(desc_key[k][0]) for x in prod]
        prod_key[k] = np.sum(prod)/(sum((desc_key[k][0])))
        #prod_key[k] = np.sum(prod)/(len(desc_key[k][0])*sum((desc_key[k][0])))
     pred.append(max(prod_key.items(), key=operator.itemgetter(1))[0])

accuracy = len([pred[i] for i in range(0, len(pred)) if pred[i] == gold[i]]) / len(pred)
print(accuracy)
