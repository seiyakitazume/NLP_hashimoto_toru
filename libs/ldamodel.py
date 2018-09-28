# ------------------
# Module import
# ------------------
import os
import time
import tqdm
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

# ------------------
# Class definition
# ------------------
class LDAModel:
    """LDAモデル計算を行うクラス."""

    def __init__(self):
        """コンストラクタ."""
        self.path_csv = None
        self.df = None
        self.topic_number = None
        self.features = None
        self.Xcv = None
        self.perplexity = None
        self.topics = None
        self.Xlda = None
        self.y_topic_id = None

    def dump(self, path_obj):
        """自身のオブジェクトを保存する."""
        with open(path_obj, "wb") as f:
            pickle.dump(self, f)
        print("Object is saved as {}".format(path_obj))

    def load(self, path_obj):
        """保存したオブジェクトを読み込む."""
        with open(path_obj, "rb") as f:
            obj = pickle.load(f)
        self.path_csv = obj.path_csv
        self.df = obj.df
        self.topic_number = obj.topic_number
        self.features = obj.features
        self.Xcv = obj.Xcv
        self.perplexity = obj.perplexity
        self.topics = obj.topics
        self.Xlda = obj.Xlda
        self.y_topic_id = obj.y_topic_id

    def prepare(self, path_csv):
        """LDA計算用のデータを準備する."""
        dfcsv = pd.read_csv(path_csv, encoding="utf8")
        self.path_csv = path_csv
        self.df = dfcsv
        cv = CountVectorizer(
            token_pattern=u'(?u)\\b\\w+\\b'
        )
        cv.fit(list(self.df["token"]))
        self.features = cv.get_feature_names()
        Xcv = cv.transform(list(self.df["token"]))
        Xcv = Xcv.toarray()
        self.Xcv = Xcv

    def cal(self, topic_number):
        """LDA計算を行う."""
        self.topic_number = topic_number
        model = LDA(
            n_components=self.topic_number,
            learning_method="batch",
            random_state=0
        )
        print("********* LDA calculation is running ... *********")
        model.fit(self.Xcv)
        self.model = model
        self.perplexity = model.perplexity(self.Xcv)
        self.topics = model.components_
        self.Xlda = model.transform(self.Xcv)
        self.y_topic_id = [np.argmax(x) for x in self.Xlda]

    def get_df_topic(self, topic_id, topn=20):
        """トピック内容をDataFrameで返す."""
        scores = self.topics[topic_id]
        words = [self.features[i] for i in range(len(scores))]
        df_topic = pd.DataFrame()
        df_topic["word"] = words
        df_topic["score"] = scores
        df_topic = df_topic.sort_values(by="score", ascending=False)
        df_topic = df_topic.head(topn)
        return df_topic

    def save_df_topic(self, topic_id, path_dir="../figs_topic/"):
        """トピック内容をグラフ化して保存する."""
        if os.path.exists(path_dir) is False:
            os.mkdir(path_dir)
        df_topic = self.get_df_topic(topic_id)
        df_topic = df_topic.sort_values(by="score", ascending=True)
        ids = [i for i in range(len(df_topic))]
        plt.figure(figsize=(12, 10))
        plt.barh(
            ids,
            df_topic["score"],
            color="blue",
            alpha=0.6
        )
        plt.yticks(ids, df_topic["word"])
        plt.xlabel("Score")
        plt.ylabel("Word")
        plt.title("LDA topic contents for topic id = {}".format(topic_id))
        path_fig = os.path.join(path_dir, "topic_id_{}.png".format(topic_id))
        plt.savefig(path_fig, bbox_inches="tight")
        plt.close()

# ------------------
# Main processing
# ------------------
if __name__=="__main__":
    path_csv = "../tokenized_data.csv"
    path_obj = "../lda_model.pickle"
    topic_number = 32
    lm = LDAModel()
    lm.prepare(path_csv)
    lm.cal(topic_number)
    lm.dump(path_obj)
    lm.load(path_obj)
    for i in range(topic_number):
        df_topic = lm.get_df_topic(i)
        print("++++++++++ For topic id = {} ++++++++++".format(i))
        print(df_topic)
        lm.save_df_topic(i, path_dir="../figs_topic/")
        print("")
    df = pd.DataFrame()
    df["filename"] = lm.df["filename"]
    for i in range(topic_number):
        df["prob_topic_{}".format(i)] = lm.Xlda[:, i]
    print(df.head())
    df.to_csv("summary.csv", encoding="utf8", index=False)
