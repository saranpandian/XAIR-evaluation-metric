import pyterrier as pt
import os
from unidecode import unidecode
os.environ["JAVA_HOME"] = "jdk-11.0.22"
import pyterrier as pt
from tqdm import tqdm
import ir_datasets
# if not pt.started():
pt.init()
# from pyterrier_colbert.indexing import ColBERTIndexer
# # import unicodedata
# import pyterrier_colbert.indexing
import pandas as pd
# from pyterrier_colbert.ranking import ColBERTFactory
# from spacy.lang.en import English
from itertools import combinations
# import spacy
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from scipy.stats import kendalltau
nlp = "NULL"
#from spaCy import *
# nlp = spacy.load('en_core_web_sm')

# from spacy.lang.en import English
import sys
# sys.stdout = open("output_log.txt", "w")
# nlp = English()
# nlp.add_pipe("sentencizer")

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
    index = pt.IndexFactory.of("./pd_index1/data.properties")
    model = pt.text.scorer(background_index = index, body_attr="text", wmodel="BM25")
    return model
  elif model=="MONOT5":
    print("loading monoT5")
    from pyterrier_t5 import MonoT5ReRanker
    model = MonoT5ReRanker()
    return model

######## class for sentence masking######
class Create_sent_masker:
  def __init__(self, nlp, neural_model, top_p,window_length, span_type):
    self.sentencizer = nlp
    self.textscorerTf = neural_model#.text_scorer()
    self.top_p = top_p
    self.window_length = window_length
    self.n_comb = 2
    self.span_type = span_type

  ########sampling the document by masking the N choose k sentences######
  def _sampler(self, doc, k, span_type):
    
    self.sample_dict = {}
    sample_dict_scores = {}
    if span_type=='sent':
      document = list(sent_tokenize(doc))
      for i, samp_sent in enumerate(document):
        self.sample_dict["s"+str(i+1)] = samp_sent
        # print(samp_sent)
        sample_dict_scores["s"+str(i+1)] = 0
    else:
      document = list(word_tokenize(doc))
      my_list = document
      start = 0
      end = len(my_list)
      step = self.window_length
      count = 0
      for i in range(start, end, step):
        x = i
        self.sample_dict["s"+str(count+1)] = " ".join(my_list[x:x+step])
        sample_dict_scores["s"+str(count+1)] = 0
        count+=1

    sample_list = []
    for n_comb in range(1,k):
        for docs in combinations(self.sample_dict.keys(), n_comb):
            temp_doc = doc
            for sent in docs:
                temp_doc = temp_doc.replace(str(self.sample_dict[sent]),"")
            sample_list.append([docs, temp_doc])
    return sample_dict_scores, sample_list
  ########scoring the document without masking for the given query######
  def actual_score(self, query, doc):
    df = pd.DataFrame( [ ["q1", query, "d1", doc],], columns=["qid", "query", "docno", "text"])
    score = self.textscorerTf.transform(df)
    return score['score'].values[0]

  ########scoring the document with masking for the given query######
  def masker(self, query, doc):
    pre_transform = []
    sample_dict_scores, sample_list = self._sampler(doc, self.n_comb+1,self.span_type)
    for i, document in enumerate(sample_list):
      pre_transform.append(["q1",query,"d{}".format(i+1),document[1],document[0]])
    df_pre_transform = pd.DataFrame(pre_transform, columns=["qid", "query", "docno", "text","span"])
    temp_text_span = df_pre_transform[["docno","span"]]
    actual_score = self.actual_score(query, doc)
    score_df = self.textscorerTf.transform(df_pre_transform)
    score_df['score'] = abs(actual_score - score_df['score'])
    return sample_dict_scores, score_df.merge(temp_text_span,on="docno")

  # def span_wise_score(colbert_df_score):

  ######## scoring the sentences/spans with fidleity scores ##############
  def pseudo_fidelity_scores(self, query, doc, model):

    # print("The query is: \n",query)
    sample_dict_scores, colbert_df_score = self.masker(query, doc)
    def length(span):
      return len(span)
    if model!='COLBERT':
    # colbert_df_score['per_sent_score'] = colbert_df_score['score']/colbert_df_score['span'].apply(length)
      colbert_df_score['per_sent_score'] = colbert_df_score['score']/colbert_df_score['span_x'].apply(length)
      colbert_df_span_list = list(colbert_df_score['span_x'])
    else:
      colbert_df_score['per_sent_score'] = colbert_df_score['score']/colbert_df_score['span'].apply(length)
      colbert_df_span_list = list(colbert_df_score['span'])
    colbert_df_score_list = list(colbert_df_score['per_sent_score'])
    # colbert_df_span_list = list(colbert_df_score['span'])

    for spans,scores in zip(colbert_df_span_list, colbert_df_score_list):
      for span in spans:
        sample_dict_scores[span] = sample_dict_scores[span]+scores
    sample_dict_scores = {k: v for k, v in sorted(sample_dict_scores.items(), key=lambda item: item[1], reverse=True)}
    temp_sents = []
    # print("The span and scores are: \n")
    # for key in sample_dict_scores.keys():
    #   print(self.sample_dict[key],"\t",sample_dict_scores[key],"\n")
    top_p = list(sample_dict_scores.keys())[:self.top_p]
    for key in top_p:
      temp_sents.append(self.sample_dict[key])
    return " ".join(temp_sents)

model_name = 'BM25'
top_explanations = 3
top_k_documents = 10
window_length = 9
span_type='word'
model = model_select(model_name)
print(model)
CRM = Create_sent_masker(nlp, model,top_explanations, window_length,span_type)

def top_k_pseudo_out(df_top_k,model):
  query_list = np.unique(df_top_k['qid'])
  final_df_pseudo = []
  for qid in query_list:
    df_temp = df_top_k[df_top_k['qid']==qid]#.rename(columns = {'text':'doc'})
    if model in ['ELECTRA','TCTCOLBERT']:
    # df_temp['pseudo_text'] = df_temp.apply(lambda x: CRM.pseudo_fidelity_scores(x['query'], x['doc']), axis=1)
      df_temp['pseudo_text'] = df_temp.progress_apply(lambda x: CRM.pseudo_fidelity_scores(x['query'], x['text'],model), axis=1)
    else:
      df_temp['pseudo_text'] = df_temp.progress_apply(lambda x: CRM.pseudo_fidelity_scores(x['query'], x['doc'],model), axis=1)
    final_df_pseudo.append(df_temp)

  df_test_top_k = pd.concat(final_df_pseudo)
  return df_test_top_k
dataset_test = ir_datasets.load('msmarco-passage/trec-dl-2020')
doc_df_qrels = []
for doc in tqdm(dataset_test.qrels_iter()):
    doc_df_qrels.append([doc.query_id,doc.doc_id, doc.relevance])
doc_df_qrels = pd.DataFrame(doc_df_qrels)
doc_df_qrels.columns = ['qid','docno','label']
# ground_truth = doc_df_qrels[doc_df_qrels['label']>=2]
print(len(np.unique(doc_df_qrels['qid'])),"length of qrels")

df_test = pd.read_csv("top_{}_files/{}_top_{}.csv".format(top_k_documents,model_name,top_k_documents))
qid_list = list(doc_df_qrels['qid'])
df_top_k_psuedo = top_k_pseudo_out(df_test[df_test['qid'].astype(str).isin(qid_list)],model=model_name)[['qid','docno','query','pseudo_text']].rename(columns = {'pseudo_text':'text'}).astype(str)
df_top_k_psuedo.to_csv("pseudo_documents/{}_top_{}_pseudo.csv".format(model_name,top_k_documents))
# df_top_k_psuedo = pd.read_csv("BM25_top10_pseudo.csv").astype(str)
print(len(np.unique(df_top_k_psuedo['qid'])),"length of queries")
df_scores = model.transform(df_top_k_psuedo)
df_scores = df_scores[['qid','docno','score','rank']]
new_df = pd.merge(df_test.astype(str), df_scores,  how='left', left_on=['qid','docno'], right_on = ['qid','docno'])

import numpy as np

doc_df_test = []
for doc in tqdm(dataset_test.queries_iter()):
    doc_df_test.append([doc.query_id,doc.text])
doc_df_test = pd.DataFrame(doc_df_test)
doc_df_test.columns = ['qid','query']
print(len(np.unique(doc_df_test['qid'])),"length of queries")
# dataset_test = ir_datasets.load('msmarco-passage/trec-dl-2020')

from pyterrier.measures import *

# Calculating Kendall Rank correlation
corr_list = []
pearson_correlation = []
for qid in np.unique(doc_df_qrels['qid']):
  temp_df_doc_psudo_doc = new_df[new_df['qid']==qid]
  # relevance = (pt.Experiment(
  #   [temp_df_doc_psudo_doc[['qid','docno','score_x']].rename(columns = {'score_x':'score'})],
  #   doc_df_test[doc_df_test['qid']==qid],
  #   doc_df_qrels[doc_df_qrels['qid']==qid],
  #   eval_metrics=[AP(rel=2)],
  #   names=["BM25"]
  #   )['AP(rel=2)'].values[0])

  # ap_score = relevance
  fidleity_corr, _ = kendalltau(temp_df_doc_psudo_doc['rank_x'], temp_df_doc_psudo_doc['rank_y'])
  corr_list.append(fidleity_corr)
  # pearson_correlation.append([qid,ap_score,fidleity_corr])
  pearson_correlation.append([qid,fidleity_corr])
print(pd.DataFrame(pearson_correlation).to_csv("correlation/{}_correlation_top{}_window_s_{}_top_{}_exp values {}.csv".format(model_name,top_k_documents,window_length,top_explanations,span_type)))
# print(pd.DataFrame(pearson_correlation)[[1,2]].corr())
print(corr_list)
print('Kendall Rank correlation: %.4f' % np.mean(corr_list))
