# -*- coding: utf-8 -*-
from keras.models import Model
from collections import defaultdict
import glob, codecs
import numpy as np
from keras import regularizers
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Dropout, Reshape, Flatten, LSTM, RepeatVector, GRU, Bidirectional
from keras.layers.wrappers import TimeDistributed
#from keras.callbacks import TensorBoard
import multiprocessing
from sklearn import metrics
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
import itertools as it
import sys
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

unique_chars = []
max_word_len = 10

def pad_word(x):
    if len(x) > max_word_len:
        return x[:max_word_len]
    else:
        return x.center(max_word_len,"0")

def wrd_to_onehot(w):
    w2d = []
    n_chars = len(unique_chars)+1
    for x in w:
        temp = n_chars*[0]
        if x == "0":
            w2d.append(temp)
        else:
            i = unique_chars.index(x)+1
            temp[i] = 1
            w2d.append(temp)
    return np.array(w2d)

def writeNexus(dm,f):
    l=len(dm)
    f.write("#nexus\n"+
                  "\n")
    f.write("BEGIN Taxa;\n")
    f.write("DIMENSIONS ntax="+str(l)+";\n"
                "TAXLABELS\n"+
                "\n")
    i=0
    for ka in dm:
        f.write("["+str(i+1)+"] '"+ka+"'\n")
        i=i+1
    f.write(";\n"+
               "\n"+
               "END; [Taxa]\n"+
               "\n")
    f.write("BEGIN Distances;\n"
            "DIMENSIONS ntax="+str(l)+";\n"+
            "FORMAT labels=left;\n")
    f.write("MATRIX\n")
    i=0
    for ka in dm:
        row="["+str(i+1)+"]\t'"+ka+"'\t"
        for kb in dm:
            row=row+str(dm[ka][kb])+"\t"
        f.write(row+"\n")
    f.write(";\n"+
    "END; [Distances]\n"+
    "\n"+
    "BEGIN st_Assumptions;\n"+
    "    disttransform=NJ;\n"+
    "    splitstransform=EqualAngle;\n"+
    "    SplitsPostProcess filter=dimension value=4;\n"+
    "    autolayoutnodelabels;\n"+
        "END; [st_Assumptions]\n")
    f.flush()

def read_data(fname):
    f = open(fname,encoding="utf-8")
    d = defaultdict(lambda: defaultdict())
    ld = defaultdict(lambda: defaultdict())
    a = f.readline().strip().split("\t")
    x = a[1:]
    lwords = []
    for line in f:
        data = line.strip().split("\t")
        lang = data[0]
        lang = lang.replace("/","_")
        lang = lang.replace("(","_")
        lang = lang.replace(")","_")
        lang = lang.replace("+","_")
        words = data[1:]
        for i, w in enumerate(words):
            if len(w) == 0:
                #print("Empty ",lang, " ", x[i-1])
                d[x[i]][lang] = "NA"
            else:
                d[x[i]][lang] = w.split(" / ")[0]
                ld[lang][x[i]] = w.split(" / ")[0]
                lwords += [z for z in w.split(" / ")]
            for c in w:
                if c not in unique_chars:
                    unique_chars.append(c)
            
        
    return d, ld, lwords

train_words = []
test_words = []
test_concepts = []

dataset = sys.argv[1]
print(dataset)

d, ld, lwords = read_data(dataset)
train_words = list(set(lwords))#Dont set if applied noisy layer
#train_words = lwords

for concept in d:
    test_concepts.append(concept)
    test_words.append(d[concept])


train_phonetic, test_phonetic = [], []
for x in train_words:
    onehotp1 = wrd_to_onehot(x)
    train_phonetic.append(onehotp1)
    
n_dim = len(unique_chars)+1
latent_dim = 8

train_phonetic = np.array(train_phonetic)

train_phonetic = sequence.pad_sequences(train_phonetic, maxlen=max_word_len)

print(train_phonetic.shape)

inputs = Input(shape=(max_word_len, n_dim))
#encoded = LSTM(latent_dim)(inputs)
encoded = Bidirectional(LSTM(latent_dim), merge_mode="concat")(inputs)

decoded = RepeatVector(max_word_len)(encoded)
decoded = LSTM(n_dim, return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(n_dim, activation="softmax"))(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(input=inputs, output=encoded)

sequence_autoencoder.compile(optimizer='adadelta', loss='categorical_crossentropy')
encoder.compile(optimizer='adadelta', loss='categorical_crossentropy')

sequence_autoencoder.summary()
sequence_autoencoder.fit(train_phonetic, train_phonetic,
                nb_epoch=5,
                batch_size=32)

dm = defaultdict(lambda: defaultdict(float))
repr_phonetic = encoder.predict(train_phonetic)
repr_dict = defaultdict()

for w, wv in zip(train_words, repr_phonetic):
    repr_dict[w] = wv
    #print(w, wv)

langs_list = []
concepts = []
for concept in d:
    print(concept)
    concepts.append(concept)
    langs = d[concept]
    langs_list += [x for x in langs]

langs_list = list(set(langs_list))

print("\n", langs_list, "\n")

fout = open(dataset+".word_vectors.txt","w")
for c in concepts:
    lang_vector, vocabulary = [], []
    for lang in langs_list:
        if concept not in d:
            continue
        if lang not in d[c]:
            continue
        w = d[c][lang]
        if w == "NA":
            continue
        #print(c, lang, w, repr_dict[w])
        word_vec = []
        for x in repr_dict[w]:
            word_vec.append(str(x))
        fout.write(c+"\t"+lang+"\t"+w+"\t"+"\t".join(word_vec)+"\n")
        lang_vector.append(repr_dict[w])
        vocabulary.append(w)
    #wv = np.array(lang_vector)
    #redu = TSNE(n_components=2, random_state=0)
    #pca = PCA(n_components=2)
    #mds = MDS(n_components=2)
    #np.set_printoptions(suppress=True)
    #Y = wv
    #Y = pca.fit_transform(wv)

    #plt.scatter(Y[:, 0], Y[:, 1])
    #for label, x, y in zip(vocabulary, Y[:, 0], Y[:, 1]):
    #    plt.annotate('$%s$' %label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    #plt.show()

#sys.exit(1)

for l1, l2 in it.combinations(langs_list, r=2):
    #print(l1, l2)
    n_concepts = 0.0
    for concept in concepts:
        if concept not in ld[l1] or concept not in ld[l2]:
            continue
        
        x1 = ld[l1][concept]
        x2 = ld[l2][concept]
        if d[concept][l1] == "NA" or d[concept][l2] == "NA":
            continue
        else:
            r1, r2 = repr_dict[x1], repr_dict[x2]
            r1, r2 = r1.flatten(), r2.flatten()
            cos_sim = np.dot(r1, r2)/(np.linalg.norm(r1)*np.linalg.norm(r2))
            #cos_sim = cos_sim/(np.linalg.norm(r1) **2 + np.linalg.norm(r2) **2 - cos_sim)
            #dm[l1][l2] += 1.0-cos_sim
            #dm[l2][l1] += 1.0-cos_sim
            dm[l1][l2] += 1.0-(cos_sim+1.0)/2.0
            dm[l2][l1] += 1.0-(cos_sim+1.0)/2.0
            n_concepts += 1.0
    dm[l1][l2] = dm[l1][l2]/n_concepts
    dm[l2][l1] = dm[l1][l2]
    
fout = codecs.open(dataset+".LSTM_concat_bidir.nex","w","utf-8")
writeNexus(dm, fout)
fout.close()

