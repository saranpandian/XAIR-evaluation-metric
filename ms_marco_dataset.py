import pandas as pd
import pyterrier as pt
import os
from tqdm import tqdm
os.environ["JAVA_HOME"] = "jdk-11.0.22"
pt.init()

import ir_datasets
dataset = ir_datasets.load("msmarco-document/trec-dl-2020")
# data = []
# for doc in tqdm(dataset.docs_iter()):
#     data.append([doc.doc_id, doc.body])
#     # df.append({'docno': doc.doc_id, 'text': doc.body}, ignore_index=True)
#     # row_to_append = pd.DataFrame([{'docno': doc.doc_id, 'text': doc.body}])
#     # df = pd.concat([df,row_to_append])
# # index the text, record the docnos as metadata
# df = pd.DataFrame(data, columns=['docno','text'])
# df.to_csv("collections.csv")
# del data
# pd_indexer = pt.DFIndexer("./pd_index")
# indexref = pd_indexer.index(df["text"], df["docno"])

qrel_data = []
for qrel in dataset.qrels_iter():
    # query_id='42255', doc_id='D1006124', relevance=0
    qrel_data.append([qrel.query_id, qrel.doc_id, qrel.relevance])
df_qrels = pd.DataFrame(qrel_data, columns=['qid','doc_id','relevance'])
df_qrels.to_csv("qrels.csv")

queries_data = []
for query in dataset.queries_iter():
    # query_id='42255', doc_id='D1006124', relevance=0
    queries_data.append([query.query_id, query.text])
df_queries = pd.DataFrame(queries_data, columns=['qid','query'])
df_queries.to_csv("queries.csv")
print(df_qrels)
print(df_queries)

index = pt.IndexFactory.of("./pd_index")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")

pt.Experiment(
    [bm25],
    df_queries,
    df_qrels,
    eval_metrics=["map", "recip_rank"]
)

