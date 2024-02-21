import pyterrier as pt
import os
from unidecode import unidecode
os.environ["JAVA_HOME"] = "jdk-11.0.22"
import pyterrier as pt
from tqdm import tqdm
import numpy as np
import ir_datasets
pt.init()
import unicodedata
import time

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
checkpoint="Checkpoint"
import pandas as pd
import inspect
import pyparsing as pp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BM25', help='one of the following models (ELECTRA, COLBERT, TCTCOLBERT,MONOT5, BM25)')
parser.add_argument('--window_length', type=int, default=3, help='Give the window size')
parser.add_argument('--num_combinations', type=int, default=2, help='number of combinations of sentence pairs in passage')

args = parser.parse_args()
# checkpoint="colbert_model_checkpoint/colbert.dnn"
# from pyterrier_colbert.ranking import ColBERTFactory
# index=("colbert_passage","index_name")

# pytcolbert = ColBERTFactory(checkpoint, *index)


from spacy.lang.en import English
nlp = English()
import pandas as pd
from itertools import combinations
import spacy
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau
nlp.add_pipe("sentencizer")
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# def divide_chunks_upon_tokenization(chunks):
#   C = {}
#   C_flag = {}
#   for i,ci in enumerate(chunks):
#     C['c'+str(i)] = ci
#     if len(tokenizer.tokenize(" ".join(ci)))>512:
#       C_flag['c'+str(i)] = False
#       # print("xxx")
#     else:
#       C_flag['c'+str(i)] = True
#   prev = ""
#   for j in C_flag.keys():
#     if prev != "":
#       C[j].insert(0, prev)
#       prev=""
#     if C_flag[j] == False:
#       prev = C[j].pop()

#   return list(C_flag.values()), list(C.values())

class sentence_scorer:

  def __init__(self, window_len, n_comb, nlp, model):
    self.model = model
    self.window_len = window_len
    self.nlp = nlp
    self.n_comb = n_comb
    self.top_p = 3
    # self.count = 0

  def sentence_score(self, query, doc):

    # chunk_list = self.nlp(doc)
    chunk_list = sent_tokenize(doc)
    temp_chunk_list = []
    # for sent in list(chunk_list.sents):
    for sent in list(chunk_list):
      temp_chunk_list.append(str(sent).strip())
    chunk_list = temp_chunk_list
    start = 0
    end = len(chunk_list)
    step = 3
    sample_list = []
    for i in range(start, end, step):
      x = i
      # sample_list.append(" ".join(chunk_list[x:x+step]))
      sample_list.append(chunk_list[x:x+step])
    # Flag = True
    # while Flag:
    #   Flag_list, chunk_list = divide_chunks_upon_tokenization(sample_list)
    #   Flag = not all(Flag_list)
    #   sample_list = chunk_list
    pre_transform = []
    for i, document in enumerate(sample_list):
      # pre_transform.append(["q1",query,"d{}".format(i+1),document])
      pre_transform.append(["q1",query,"d{}".format(i+1)," ".join(document)])
    df_pre_transform = pd.DataFrame(pre_transform, columns=["qid", "query", "docno", "text"])
    # self.count+=1
    # print(self.count)
    df_scored = self.model.transform(df_pre_transform)
    return df_scored['score'].max()

def model_select(model):
  if model=="ELECTRA":
    print("loading electra")
    import pyterrier_dr
    model = pyterrier_dr.ElectraScorer()
    return model
  elif model=="TCTCOLBERT":
    print("loading tctCOLBERT")
    import pyterrier_dr
    model = pyterrier_dr.TctColBert()
    return model
  elif model=='COLBERT':
    print("loading colbert")
    import pyterrier_colbert.indexing
    from pyterrier_colbert.ranking import ColBERTFactory
    index=("colbert_passage","index_name")
    checkpoint="http://www.dcs.gla.ac.uk/~craigm/ecir2021-tutorial/colbert_model_checkpoint.zip"
    model = ColBERTFactory(checkpoint, *index).text_scorer()
    return model
  elif model=='BM25':
    print("loading BM25")
    index = pt.IndexFactory.of("./passage_index/data.properties")
    model = pt.text.scorer(background_index = index, body_attr="text", wmodel="BM25")
    return model
  elif model=="MONOT5":
    print("loading monoT5")
    from pyterrier_t5 import MonoT5ReRanker
    model = MonoT5ReRanker()
    return model
# pytcolbert = monoT5
# textscorerTf = pytcolbert.text_scorer()
model_name = args.model
model = model_select(model_name)

CRM = sentence_scorer(window_len=args.window_length, n_comb=args.num_combinations, nlp=nlp, model=model)
print("-------------loading sentences-----------------")
df = pd.read_csv("document_level/BM25_with_docs_1000.csv")#.head(10)

dataset_test = ir_datasets.load('msmarco-document/trec-dl-2020')
doc_df_qrels = []
for doc in tqdm(dataset_test.qrels_iter()):
    doc_df_qrels.append([doc.query_id,doc.doc_id, doc.relevance])

doc_df_qrels = pd.DataFrame(doc_df_qrels)#.astype(str)
doc_df_qrels.columns = ['qid','docno','label']

qid_list = list(doc_df_qrels['qid'].astype(int))
df = df[df['qid'].astype(int).isin(qid_list)]

print("-------------scoring sentences-----------------")
start_time = time.time()
df['score'] = df.progress_apply(lambda x: CRM.sentence_score(x['query'], x['text']), axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

print(df.head(5))
print(df.to_csv("document_level/{}_with_docs_1000.csv".format(model_name)))
