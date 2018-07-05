import sys
import csv
from time import sleep
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
import sent2vec
import torch
import torch.nn as nn
from torch.autograd import Variable

f = open('emails_dataset/emails.dataset','r')

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

lis = f.readlines()

emails_train = []
emails_inference = []
gold_train = []
gold_inference = []

concepts_train = {'REMINDER':[], 'HUMOR' : [], 'EVENT': [], 'EMPLOYEE': [], 'MEETING' : []}
concepts_inference = {'POLICY' : [], 'CONTACT' : []}
concepts = {'REMINDER':[], 'HUMOR' : [], 'EVENT': [], 'EMPLOYEE': [], 'MEETING' : [], 'POLICY' : [], 'CONTACT' : []}
concepts_dic = {'REMINDER':3, 'HUMOR' : 4, 'EVENT': 5, 'EMPLOYEE': 6, 'MEETING' : 7, 'POLICY' : 1, 'CONTACT' : 2}

TRAIN = [3,4,5,6,7]
INFERENCE = [1,2]

for each in lis:
    item = each.split("\t")
    item_email =  ' '.join(e for e in item[2:])
    item_email = item_email.replace('</br>',' ').replace(':',' ')
    if concepts_dic[item[1]] in TRAIN:
        gold_train.append(concepts_dic[item[1]]-2) 
        emails_train.append(item_email)
    else:
        gold_inference.append(concepts_dic[item[1]])
        emails_inference.append(item_email)
    #emails_len.append(len(item_email))

#max_words_email = max(emails_len)
#total_email = len(emails_train)

embedding_size = 700
all_mail_vec = []

tknzr = StanfordTokenizer(SNLP_TAGGER_JAR, encoding='utf-8')

s = ' <delimiter> '.join(emails_train)
tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
emails_train = tokenized_sentences_SNLP[0].split(' <delimiter> ')
embs_email_train = model.embed_sentences(emails_train)  

s = ' <delimiter> '.join(emails_inference)
tokenized_sentences_SNLP = tokenize_sentences(tknzr, [s])
emails_inference = tokenized_sentences_SNLP[0].split(' <delimiter> ')
embs_email_inference = model.embed_sentences(emails_inference)

f.close()

f = open('emails_dataset/emailExplanations_Dec23.sorted.txt','r')


#TRAIN = 3 4 5 6 7

#154 #1 #REMINDER 35
#134 #2 #HUMOR 20
#138 #3 #EVENT 30
#149 #4 #EMPLOYEE 25
#142 #5 #MEETING 25
#146 #6 #POLICY 25
#167 #7 #CONTACT 25
 
lis = f.readlines()

for each in lis:
    item = each.split("\t")
    t1 = item[2].split("_")
    item_email =  ' '.join(e for e in item[3:])
    item_email = item_email.replace('</br>',' ').replace(':',' ')
    concepts[item[1]].append((int(t1[0]),item_email))

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

embs_desc_train = {}
embs_desc_inference = {}

for k,v in concepts.items():
        if concepts_dic[k] in TRAIN:
                embs_desc_train[k] = model.embed_sentences(desc_key[k][1])
        else:
                embs_desc_inference[k] = model.embed_sentences(desc_key[k][1])

#embs_desc_train = list(chain.from_iterable(embs_desc_train))
#embs_desc_inference = list(chain.from_iterable(embs_desc_inference))

#embs_desc_train = np.vstack(embs_desc_train)
#embs_desc_inference = np.vstack(embs_desc_inference)

embs_email_train = np.vstack(embs_email_train)
embs_email_inference = np.vstack(embs_email_inference)

total_email_train = len(embs_email_train)
total_email_inference = len(embs_email_inference)


#train, test, label_train, label_test = train_test_split(final_all, gold_new, test_size = 0.1)
#trainin_desc_new = np.vstack(embs_desc_new), test = x[:80,:], x[80:,:]order = np.arange(train.shape[0])

order = np.arange(total_email_train)
np.random.shuffle(order)

gold_train = np.array(gold_train)[order]
embs_email_train = embs_email_train[order]

split = int(total_email_train/2)
train, test = embs_email_train[:split,:], embs_email_train[split:,:]
label_train, label_test = np.array(gold_train)[:split], np.array(gold_train)[split:]

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()

        #self.linear = nn.Linear(input_size, input_size)
        self.weight = torch.nn.Parameter(torch.zeros(1, input_size))
        self.tan = torch.nn.Tanh()
    def forward(self, desc, train, train_keys):

        result = []

        for k in train_keys:
            out = desc[k]*self.weight
            #out = self.tan(out)
            out = torch.mm(out,train.permute(1,0))
            
            result.append(torch.sum(out,0))

        result = torch.stack(result).squeeze(1).permute(1,0)
            #.unsqueeze(0))
            #desc[k]*train[0].data            
            #out = desc[k]*torch.unsqueeze(train,dim=0)
            #out = out * train
        return result

max_epoch = 5000
learning_rate = 0.001

train_keys = []
for k,v in embs_desc_train.items():
	train_keys.append(k)

inf_keys = []
for k,v in embs_desc_inference.items():
        inf_keys.append(k)

print(train_keys)
print(inf_keys)
classes = len(TRAIN)

model = LogisticRegression(embedding_size, classes)

criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

new_desc = {}
for k in train_keys:
    new_desc[k] = Variable(torch.from_numpy(embs_desc_train[k]))

new_desc_inf = {}
for k in inf_keys:
    new_desc_inf[k] = Variable(torch.from_numpy(embs_desc_inference[k]))

prev = 100

test_input = Variable(torch.from_numpy(test))
label_test_input = torch.from_numpy(np.array(label_test)).long()

inf_input = Variable(torch.from_numpy(embs_email_inference))
label_inf_input = torch.from_numpy(np.array(gold_inference)).long()

for epoch in range(max_epoch):
    model.train()
    order = np.arange(train.shape[0])
    np.random.shuffle(order)

    label_train = np.array(label_train)[order]
    train = train[order]

    train_input = Variable(torch.from_numpy(train))
    label_train_input = Variable(torch.from_numpy(label_train))

    optimizer.zero_grad()
    output = model(new_desc, train_input, train_keys)
    loss = criterion(output, label_train_input-1)
    loss.backward()
    optimizer.step()
    print(epoch, loss.data[0])    

    model.eval()

    test_input = Variable(torch.from_numpy(test))
    label_test_input = torch.from_numpy(np.array(label_test)).long()

    correct = 0
    optimizer.zero_grad()
    output_test = model(new_desc, test_input, train_keys)
    loss_test = criterion(output_test, Variable(label_test_input-1))

    _, predicted = torch.max(output_test.data, 1)
    correct += (predicted == label_test_input-1).sum()

    print("Test Accuracy", correct/len(label_test_input), "Test Loss", loss_test.data[0])

    if (prev - loss_test.data[0]) < 0 and (prev - loss_test.data[0]) < 0.001:
        print("-----")
        print("FINAL")

        correct = 0

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label_train_input.data.long()-1).sum()
        print("Train Accuracy", correct/len(label_train_input))

        correct = 0
        optimizer.zero_grad()
        output = model(new_desc_inf, inf_input, inf_keys)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label_inf_input-1).sum()

        print("Inference Accuracy", correct/len(label_inf_input))
        print("-----")
        exit()

    prev = loss_test.data[0]
    #embs_desc_train
    if epoch%20 == 0:
        print("-----")     
        correct = 0	

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label_train_input.data.long()-1).sum()
        print("Train Accuracy", correct/len(label_train_input))

        correct = 0
        optimizer.zero_grad()
        output = model(new_desc, test_input, train_keys)
       
        loss_test = criterion(output, Variable(label_test_input-1))

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label_test_input-1).sum()

        print("Test Accuracy", correct/len(label_test_input), "Test Loss", loss_test.data[0])
 
        correct = 0
        optimizer.zero_grad()
        output = model(new_desc_inf, inf_input, inf_keys)
        _, predicted = torch.max(output.data, 1)
        correct += (predicted == label_inf_input-1).sum()

        print("Inference Accuracy", correct/len(label_inf_input))
        print("-----")

optimizer.zero_grad()
output = model(new_desc_inf, inf_input, inf_keys)
_, predicted = torch.max(output.data, 1)

new_predicted = []
new_gold_inference = []
rev_dic = {'0':'POLICY', '1':'CONTACT'}

for i in range(len(predicted)):
        new_predicted.append(rev_dic[str(predicted[i])])
        new_gold_inference.append(rev_dic[str(gold_inference[i]-1)])


rows = zip(emails_inference, new_gold_inference, new_predicted)        #sleep(1)
with open('bi_not_weighted_logistic.csv', 'w') as f:
        writer = csv.writer(f)
        for row in rows:
                writer.writerow(row)
'''
coss = crierion(output, label_train_input-1)
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
