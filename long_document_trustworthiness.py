import pandas as pd
import pyterrier as pt
from tqdm import tqdm
import ir_datasets
import re
import os
from unidecode import unidecode
os.environ["JAVA_HOME"] = "jdk-11.0.17"
from pyterrier_colbert.indexing import ColBERTIndexer
# import unicodedata
import pyterrier_colbert.indexing
# from spacy.lang.en import English
from itertools import combinations
import spacy
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau
#from spaCy import *
# nlp = spacy.load('en_core_web_sm')
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from spacy.lang.en import English
import sys
import random
pt.init()
# sys.stdout = open("output_log.txt", "w")
nlp = English()
nlp.add_pipe("sentencizer")
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string
import pyparsing as pp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='BM25', help='one of the following models (ELECTRA, COLBERT, TCTCOLBERT,MONOT5, BM25)')
parser.add_argument('--top_explanations', type=int, default=3, help='Give the window size')
parser.add_argument('--window_length', type=int, default=2, help='number of combinations of sentence pairs in passage')
parser.add_argument('--span_type', type=str, default='word', help = 'if the evaluation is at word level or sentence level('word','sentence')')

args = parser.parse_args()


def clean_string(text):
    text = ''.join([word for word in text if word not in string.punctuation])
    text = text.lower()
    #text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def cosine_sim_vectors(vec1,vec2):
    vec1 = vec1.reshape(1,-1)
    vec2 = vec2.reshape(1,-1)
    return cosine_similarity(vec1,vec2)[0][0]

def similarity(sentence1, sentence2):
    # if len(sentences)!=2:
    #     return 'Error'
    cleaned = list(map(clean_string, [sentence1, sentence2]))
    vectorizer = CountVectorizer().fit_transform(cleaned)
    vectors = vectorizer.toarray()
    csim = cosine_similarity(vectors)
    # lev = Levenshtein.distance(cleaned[0],cleaned[1])
    cos = cosine_sim_vectors(vectors[0],vectors[1]) *100
    # lev2 = Levenshtein.distance(cleaned2[0],cleaned2[1],)
    return cos



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
# pytcolbert = textscorerTf
# textscorerTf = pytcolbert
######## class for sentence masking######
class sentence_scorer:

  def __init__(self, window_len, n_comb, nlp, model):
    self.model = model
    self.window_len = window_len
    self.nlp = nlp
    self.n_comb = n_comb
    self.top_p = 3
  def _sentence_score(self, query, doc):

    chunk_list = self.nlp(doc)
    temp_chunk_list = []
    for sent in list(chunk_list.sents):
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
      pre_transform.append(["q1",query,"d{}".format(i+1)," ".join(document)])
    df_pre_transform = pd.DataFrame(pre_transform, columns=["qid", "query", "docno", "text"])
    df_scored = self.model(df_pre_transform)
    return df_scored['score'].max()

  ########sampling the document by masking the N choose k sentences######
  def _sampler(self, doc, k, span_type):
    document = nlp(doc)
    self.sample_dict = {}
    sample_dict_scores = {}
    sample_dict_count = {}
    if span_type=='sent':
      for i, samp_sent in enumerate(list(document.sents)):
        self.sample_dict["s"+str(i+1)] = samp_sent
        sample_dict_scores["s"+str(i+1)] = 0
        sample_dict_count["s"+str(i+1)] = 0
    else:
      my_list = document
      start = 0
      end = len(my_list)
      step = 5
      count = 0
      for i in range(start, end, step):
        x = i
        self.sample_dict["s"+str(count+1)] = my_list[x:x+step]
        sample_dict_scores["s"+str(count+1)] = 0
        sample_dict_count["s"+str(count+1)] = 0
        count+=1

    sample_list = []
    d_len = len(self.sample_dict.keys())
    print("length of documents: ",d_len)
    comb_len = len([x for x in combinations(self.sample_dict.keys(), 2)])
    # samples_needed = int((d_len)*2)
    # if samples_needed>500:
    #   samples_needed = 500
    if d_len>1000 or comb_len>1000:
      flag_samp = 1000
      for n_comb in range(1000):
              # if d_len>3:
              if d_len<200:
                docs = random.sample(list(self.sample_dict.keys()),k=k+20)
              else:
                docs = random.sample(list(self.sample_dict.keys()),k=int(d_len/2)+k)
              temp_doc = doc
              # if d_len!=1:
              for sent in docs:
                    temp_doc = temp_doc.replace(str(self.sample_dict[sent]),"")
              sample_list.append([docs, temp_doc])
    elif d_len<=1000 or comb_len<=1000:
      flag_samp = 1
      for n_comb in range(1,k):
          for docs in combinations(self.sample_dict.keys(), n_comb):
              # else:
              #   docs = [list(self.sample_dict.keys())[n_comb]]
              # print(docs)
              # print(self.sample_dict)
              # print(docs)
              temp_doc = doc
              # if d_len!=1:
              for sent in docs:
                    temp_doc = temp_doc.replace(str(self.sample_dict[sent]),"")
              sample_list.append([docs, temp_doc])
    elif d_len==1:
      sample_list.append([list(self.sample_dict.keys()),doc])
    else:
      for samp in range(d_len):
        docs = [list(self.sample_dict.keys())[samp]]
        temp_doc = doc
        for sent in docs:
          temp_doc = temp_doc.replace(str(self.sample_dict[sent]),"")
        sample_list.append([docs, temp_doc])
    return sample_dict_scores, sample_dict_count, sample_list, flag_samp

  ########scoring the document with masking for the given query######
  def masker(self, query, doc):

    pre_transform = []
    sample_dict_scores, sample_dict_count, sample_list, sample_len = self._sampler(doc, self.n_comb+1,'sent')
    for i, document in enumerate(sample_list):
      pre_transform.append(["q1",query,"d{}".format(i+1),document[1],document[0]])
    df_pre_transform = pd.DataFrame(pre_transform, columns=["qid", "query", "docno", "text","span"])
    temp_text_span = df_pre_transform[["docno","span"]]
    actual_score = self._sentence_score(query, doc)
    # score_df = self.textscorerTf.transform(df_pre_transform[["qid", "query", "docno", "text"]])
    df_pre_transform['score'] = df_pre_transform.progress_apply(lambda x: self._sentence_score(x['query'], x['text']), axis=1)
    # print(score_df)
    df_pre_transform['score'] = actual_score - df_pre_transform['score']
    return sample_dict_scores, sample_dict_count, df_pre_transform.merge(temp_text_span,on="docno"), sample_len

  # def span_wise_score(colbert_df_score):

  ######## scoring the sentences/spans with fidleity scores ##############
  def top_p_span_list(self, qid, query, docid, doc):

    # print("The query is: \n",query)
    sample_dict_scores, sample_dict_count, colbert_df_score, sample_len = self.masker(query, doc)
    def length(span):
      return len(span)
    # colbert_df_score['per_sent_score'] = colbert_df_score['score']/colbert_df_score['span'].apply(length)
    colbert_df_score['per_sent_score'] = colbert_df_score['score']/colbert_df_score['span_x'].apply(length)
    colbert_df_score = colbert_df_score.sort_values(by = 'score', ascending = False)
    colbert_df_score_list = list(colbert_df_score['per_sent_score'])
    # colbert_df_span_list = list(colbert_df_score['span'])
    colbert_df_span_list = list(colbert_df_score['span_x'])
    ######################################################################################################################################
    for spans,scores in zip(colbert_df_span_list[:5], colbert_df_score_list[:5]):
      for span in spans:
        if sample_len==1:
          sample_dict_count[span] = sample_dict_count[span]+scores
        elif sample_len==1000:
          sample_dict_count[span] = sample_dict_count[span]+1
    sample_dict_scores = {k: v for k, v in sorted(sample_dict_count.items(), key=lambda item: item[1], reverse=True)}
    ######################################################################################################################################
    ######################################################################################################################################
    # for spans,scores in zip(colbert_df_span_list, colbert_df_score_list):
    #   for span in spans:
    #     sample_dict_scores[span] = sample_dict_scores[span]+scores
    #     sample_dict_count[span] = sample_dict_count[span]+1
    # for span in sample_dict_scores.keys():
    #   if sample_dict_count[span]!=0:
    #     sample_dict_scores[span] = sample_dict_scores[span]/sample_dict_count[span]
    #   else:
    #     pass
    # sample_dict_scores = {k: v for k, v in sorted(sample_dict_scores.items(), key=lambda item: item[1], reverse=True)}
    ########################################################################################################################################

    temp_sents1 = []
    temp_sents = []
    # print("The span and scores are: \n")
    # for key in sample_dict_scores.keys():
    #   print(self.sample_dict[key],"\t",sample_dict_scores[key],"\n")
    top_p = list(sample_dict_scores)[:self.top_p]
    print(top_p)
    for key in top_p:
      temp_sents1.append([qid,docid,key,str(self.sample_dict[key])])
      temp_sents.append(str(self.sample_dict[key]))
    return pd.DataFrame(temp_sents1, columns=["qid", "docno","sid", "span"]), " ".join(temp_sents), sample_len
  # def pseudo_fidelity_scores(self, query, doc):

  #   # print("The query is: \n",query)
  #   sample_dict_scores, colbert_df_score = self.masker(query, doc)

  #   def length(span):
  #     return len(span)
  #   colbert_df_score['per_sent_score'] = colbert_df_score['score']/colbert_df_score['span_x'].apply(length)
  #   # colbert_df_score['per_sent_score'] = colbert_df_score['score']/colbert_df_score['span'].apply(length)
  #   colbert_df_score_list = list(colbert_df_score['per_sent_score'])
  #   colbert_df_span_list = list(colbert_df_score['span_x'])
  #   # colbert_df_span_list = list(colbert_df_score['span'])
  #   for spans,scores in zip(colbert_df_span_list, colbert_df_score_list):
  #     for span in spans:
  #       sample_dict_scores[span] = sample_dict_scores[span]+scores
  #   sample_dict_scores = {k: v for k, v in sorted(sample_dict_scores.items(), key=lambda item: item[1], reverse=True)}

  #   temp_sents = []
  #   # print("The span and scores are: \n")
  #   # for key in sample_dict_scores.keys():
  #   #   print(self.sample_dict[key],"\t",sample_dict_scores[key],"\n")
  #   top_p = list(sample_dict_scores.keys())[:self.top_p]
  #   for key in top_p:
  #     temp_sents.append(str(self.sample_dict[key]))
  #   return " ".join(temp_sents)
model_name = args.model
top_explanations = args.top_explanations
top_k_documents = args.top_k
window_length = args.window_length
span_type= args.span_type
model = model_select(model_name)
print(model)

CRM = sentence_scorer(window_len=3, n_comb=1, nlp=nlp, model=model)

def top_k_trust_out(df_top_k):
  query_list = np.unique(df_top_k['qid'])
  final_df_pseudo = []
  final_df_pseudo1 = []
  len_of_samples = []
  # query_list = query_list[1:].copy()
  for qid in query_list:

    df_temp = df_top_k[df_top_k['qid']==qid]
    temp_span_list = []
    temp_len_of_samples = []
    for query, docno, text in zip(df_temp['query'],df_temp['docno'],df_temp['text']):
        temp1, temp, sample_len = CRM.top_p_span_list(qid, query, docno, text)
        temp_len_of_samples.append(sample_len)
        final_df_pseudo.append(temp1)
        temp_span_list.append(temp)
    len_of_samples.append(temp_len_of_samples)
    df_temp['pseudo_text'] = temp_span_list
    final_df_pseudo1.append(df_temp)
  # final_df_pseudo1.append(df_temp)
  df_test_top_k_trust = pd.concat(final_df_pseudo)
  df_test_top_k_fidelity = pd.concat(final_df_pseudo1)
  return df_test_top_k_trust, df_test_top_k_fidelity, len_of_samples

# def top_k_pseudo_out(df_top_k):
#   query_list = np.unique(df_top_k['qid'])
#   final_df_pseudo = []
#   for qid in query_list:
#     print(qid)
#     df_temp = df_top_k[df_top_k['qid']==qid]
#     df_temp['pseudo_text'] = df_temp.progress_apply(lambda x: CRM.pseudo_fidelity_scores(x['query'], x['text']), axis=1)
#     final_df_pseudo.append(df_temp)

#   df_test_top_k = pd.concat(final_df_pseudo)
#   return df_test_top_k
dataset_test = ir_datasets.load('msmarco-document/trec-dl-2020')
doc_df_qrels = []
for doc in tqdm(dataset_test.qrels_iter()):
    doc_df_qrels.append([doc.query_id,doc.doc_id, doc.relevance])

doc_df_qrels = pd.DataFrame(doc_df_qrels)#.astype(str)
doc_df_qrels.columns = ['qid','docno','label']

qid_list = list(doc_df_qrels['qid'].astype(int))

df_test = pd.read_csv("document_level/BM25_top_10.csv")#.head(100)


df_test_top_k_trust, df_top_k_psuedo, len_of_samples = top_k_trust_out(df_test[df_test['qid'].astype(int).isin(qid_list)])
print(len_of_samples)
df_top_k_psuedo = df_top_k_psuedo[['qid','docno','query','pseudo_text']].rename(columns = {'pseudo_text':'text'}).astype(str)

df_scores = model.transform(df_top_k_psuedo)
# df_scores.to_csv("pseudo_scores2.csv")
df_scores = df_scores[['qid','docno','score']].sort_values(by = ['qid', 'score'], ascending = [True, False], na_position = 'first')
new_df = pd.merge(df_scores.astype(str), df_test.astype(str), how='left', left_on=['qid','docno'], right_on = ['qid','docno'])
# new_df.to_csv("pseudo_scores2.csv")
corr_list = []
for qid in np.unique(new_df['qid']):
  temp_df_doc_psudo_doc = new_df[new_df['qid']==qid].sort_values(by = ['score_y'], ascending = [False])
  print(temp_df_doc_psudo_doc['score_x'], temp_df_doc_psudo_doc['score_y'])
  corr, _ = kendalltau(np.argsort(temp_df_doc_psudo_doc['score_y'])[::-1], np.argsort(temp_df_doc_psudo_doc['score_x'])[::-1])
  corr_list.append(corr)
print(corr_list)
print('Kendall Rank correlation: %.5f' % np.mean(corr_list))


# doc_rel_qrels = doc_df_qrels[doc_df_qrels['label']>=2]

# # df_test = pd.read_csv("colbert_top_10.csv").head(100).astype(str)
# df_test = df_test.astype(str)
# df_topk_rel_docs = df_test.merge(doc_rel_qrels,on=['qid','docno'])
# print(df_topk_rel_docs)
# df_top_k_psuedo = top_k_trust_out(df_topk_rel_docs)#[['qid','docno','pseudo_text']].rename(columns = {'pseudo_text':'text'})
# df_test_top_k_trust.to_csv("document_level/BM25_top_trust_10.csv")
df_documents = pd.read_csv("document_level/trustworthiness.csv").astype(str)
df_documents.rename(columns = {'docid':'docno1'},inplace=True)#['docno','text']
df_documents.rename(columns = {'docno':'docid'},inplace=True)#['docno','text']
df_documents.rename(columns = {'docno1':'docno'},inplace=True)#['docno','text']

df_top_k_rel_text_spans = (df_test_top_k_trust.astype(str).merge(df_documents,on='docno'))

df_top_k_rel_text_spans['match scores'] = df_top_k_rel_text_spans.apply(lambda x: similarity(x['span'], x['text'])/100, axis=1)
# print(df_top_k_rel_text_spans.to_csv("document_level/trustworthiness_pass_scores.csv"))
df_agg_top_k_rel_text_spans_sid = (df_top_k_rel_text_spans.groupby(['qid','docno',"sid"])['match scores'].max())

df_agg_top_k_rel_text_spans_docid = (df_agg_top_k_rel_text_spans_sid.groupby(['qid','docno']).mean())
df_agg_top_k_rel_text_spans_qid = df_agg_top_k_rel_text_spans_docid.groupby(['qid']).mean()

trustworthiness_score = df_agg_top_k_rel_text_spans_qid.mean()
print('Trustworthiness correlation: %.5f' % trustworthiness_score)
#df_top_k_psuedo.to_csv("")
# print("loading maps")
# df = pd.read_csv("data",sep = '\t', header=None).astype(str)
# df.columns = ['docno','docid']
# print("loading passages")
# df_passages = pd.read_csv("collection.tsv",sep='\t').astype(str)
# df_passages.columns = ['docno','text']
# # df = df.head(100)
# print("merge start")
# print(df_passages.head(100))
# df_merge = df.merge(df_passages,on='docno')
# print(len(df_merge))
# print("start grouping")
# df_temp = df_merge.sort_values(by = ['docid','docno'])#.apply(' '.join)
# df_temp.to_csv("document_level/trustworthiness.csv")
# print(len(df_temp))


# dataset_test = ir_datasets.load('msmarco-document/trec-dl-2020')
# df = pd.read_csv("document_level/trustworthiness.csv")
# print(df.head(10))
# doc_df_qrels = []
# for doc in tqdm(dataset_test.qrels_iter()):
#     doc_df_qrels.append([doc.query_id,doc.doc_id, doc.relevance])
# doc_df_qrels = pd.DataFrame(doc_df_qrels)
# doc_df_qrels.columns = ['qid','docno','label']
