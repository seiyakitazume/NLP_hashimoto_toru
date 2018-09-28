# ----------------------
# Module import
# ----------------------
import re
import os
import time
import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from libs.tokendata import TokenData
from libs.ldamodel import LDAModel

# -----------------------
# Function definition
# -----------------------

def get_dfcsv(path_csv):
    """csvファイルを読み込みDataFrameで返す関数."""
    return pd.read_csv(path_csv, encoding="utf8")

def get_date(filename):
    """ファイル名から日付を抽出して返す."""
    pattern = re.compile(r"\d+年\d+月\d+日")
    obj = re.search(pattern, filename)
    if obj is not None:
        date = obj[0]
    else:
        date = ""
    return date

def get_vol(filename):
    """ファイル名からvolume numberを抽出して返す."""
    pattern = re.compile(r"Vol.\d+|vol.\d+")
    obj = re.search(pattern, filename)
    if obj is not None:
        try:
            vol = obj[0].split(".")[1]
            vol = int(vol)
        except:
            vol = -1
    else:
        vol = -1
    return vol

def make_tokenized_data(path_dir, path_csv):
    """dataディレクトリのテキストファイルから学習用のデータを生成しcsvファイルで保存する."""
    td = TokenData(path_dir)
    td.do(path_csv)
    print("Tokenized text data is saved as {}".format(path_csv))

def build_lda_model(path_csv, path_obj, topic_number):
    """LDAモデルを構築保存した上そのオブジェクトを返す."""
    lm = LDAModel()
    lm.prepare(path_csv)
    lm.cal(topic_number)
    lm.dump(path_obj)
    return lm

def read_lda_model(path_obj):
    """構築したLDAモデルのオブジェクトを読み込み返す."""
    lm = LDAModel()
    lm.load(path_obj)
    return lm

def get_topic_contents(lm, path_figs_topic):
    """LDAモデルのトピック内容をグラフで作成して保存する."""
    topic_ids = [i for i in range(lm.topic_number)]
    for topic_id in topic_ids:
        print("********** Topic contents for topic_id = {} **********".format(topic_id))
        df_topic = lm.get_df_topic(topic_id)
        lm.save_df_topic(topic_id, path_figs_topic)
        print(df_topic)
        print("")

def get_perplexity_dependence(path_csv, path_fig_perplexity):
    """トピック数とperplexityの関係をグラフで保存する."""
    print("********** Calculation for perplexity dependence on topic number is running ... **********")
    topic_numbers = [2, 4, 8, 16, 24, 32, 40, 48, 64]
    pps = []
    for topic_number in topic_numbers:
        lm = LDAModel()
        lm.prepare(path_csv)
        lm.cal(topic_number)
        pps.append(lm.perplexity)
    plt.figure(figsize=(10, 10))
    plt.scatter(
        topic_numbers,
        pps,
        color="blue",
        alpha=0.6,
        s=200
    )
    plt.plot(
        topic_numbers,
        pps,
        color="blue"
    )
    plt.xlabel("Topic number")
    plt.xscale("log")
    plt.ylabel("LDA perplexity")
    plt.title("Perplexity dependence on topic_number")
    plt.savefig(path_fig_perplexity)
    print("Perplexity dependence is saved as {}".format(path_fig_perplexity))

def get_report(lm, path_report):
    """LDAモデルのトピック所属確率に関するレポートを出力する."""
    df_report = pd.DataFrame()
    df_report["filename"] = lm.df["filename"]
    df_report["volume"] = [get_vol(filename) for filename in lm.df["filename"]]
    df_report["date"] = [get_date(filename) for filename in lm.df["filename"]]
    topic_ids = [i for i in range(lm.topic_number)]
    for topic_id in topic_ids:
        df_report["prob_topic_id_{}".format(topic_id)] = lm.Xlda[:, topic_id]
    df_report = df_report.sort_values(by="volume", ascending=True)
    df_report.to_csv(path_report, encoding="utf8", index=False)
    print("Probability for each topic result is saved as {}".format(path_report))

def get_figs_topic_prob(path_report, path_figs_topic_prob):
    """volume番号にトピック所属確率をトピックごとに出力する."""
    if os.path.exists(path_figs_topic_prob) is False:
        os.mkdir(path_figs_topic_prob)
    df = get_dfcsv(path_report)
    cols = df.columns[3:]
    for col in cols:
        print("******** Output topic probability graph for {} ********".format(col))
        plt.figure(figsize=(10, 10))
        plt.scatter(
            df["volume"],
            df[col],
            color="blue",
            alpha=0.6,
            s=200
        )
        plt.plot(
            df["volume"],
            df[col],
            color="blue"
        )
        plt.xlabel("Volume")
        plt.ylabel("Probability")
        plt.title("Topic probability for {}".format(col.replace("prob_", "")))
        path_save = os.path.join(path_figs_topic_prob, col + ".png")
        plt.savefig(path_save)
        plt.close()


# -----------------------
# Settings
# -----------------------
path_dir = "./data/"
path_csv = "./tokenized_data.csv"
path_obj = "lda_obj.pickle"
path_figs_topic = "figs_topic/"
path_figs_topic_prob = "figs_topic_prob/"
path_fig_perplexity = "topic_number_dependence.png"
path_report = "./lda_topic_prob_report.csv"

# -----------------------
# Main processing
# -----------------------
if __name__=="__main__":
    make_tokenized_data(path_dir, path_csv)
    get_perplexity_dependence(path_csv, path_fig_perplexity)
    lm = build_lda_model(path_csv, path_obj, topic_number=32)
    lm = None
    lm = read_lda_model(path_obj)
    get_topic_contents(lm, path_figs_topic)
    get_report(lm, path_report)
    get_figs_topic_prob(path_report, path_figs_topic_prob)
