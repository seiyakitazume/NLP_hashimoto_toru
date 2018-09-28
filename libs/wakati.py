# -------------------------------
# モジュールインポート部分
# -------------------------------
import os
import time
import tqdm
import numpy as np
import pandas as pd
import MeCab

# ---------------------------------
# クラス定義や関数定義の部分
# ---------------------------------

# 単語リストを半角空白で結合された1行テキストに変換する関数
def list2line(words_list):
    line = ""
    for word in words_list:
        line += word
        line += " "
    return line[:-1]

# 分かち書き用のクラス定義
class Wakati:
    # コンストラクタの定義
    def __init__(self, text):
        self.text = text
        self.tokens = None
        self.targets = ["名詞", "動詞", "形容詞"]
        self.stopwords = ["する", "もの", "れる", "ない", "られる", "こと", "ある", "ため", "これ", "いる", "なる", "よる", "よう"]

    # 分かち書きを行うメソッド
    def tokenize(self):
        words = self.get_words()
        self.tokens = self.get_stopped_words(words)
        return self.tokens

    # テキストの形態素解析結果をDataFrameで返す関数
    def get_dfw(self):
        t = MeCab.Tagger("Owakati")
        t.parse("")
        node = t.parseToNode(self.text)

        surfaces = []
        stems = []
        poss = []

        while node:
            surface = node.surface
            feature = node.feature.split(",")
            stem = feature[6]
            pos = feature[0]
            surfaces.append(surface)
            stems.append(stem)
            poss.append(pos)
            node = node.next

        df = pd.DataFrame()
        df["SURFACE"] = surfaces[1:-1]
        df["STEM"] = stems[1:-1]
        df["POS"] = poss[1:-1]
        return df

    # 形態素解析から対象となる品詞の単語リストを返す関数
    def get_words(self):
        df = self.get_dfw()
        words = []
        for row in df.iterrows():
            for target_pos in self.targets:
                if row[1]["POS"] == target_pos:
                    if row[1]["STEM"] != "*":
                        words.append(row[1]["STEM"])
        return words

    # 単語リストにストップワードを適用した結果を返す関数
    def get_stopped_words(self, words):
        stopped_words = [word for word in words if word not in self.stopwords]
        return stopped_words



# ----------------------------------------
# メインの処理
# ---------------------------------------
if __name__ == "__main__":
    # 原子データの読み込み
    path = "../extract_atoms_data.csv"
    df = pd.read_csv(path, encoding="utf-8")
    texts = df["TEXT"]
    # 分かち書きテキストデータ列の計算
    tokens = [list2line(Wakati(text).tokenize()) for text in tqdm.tqdm(texts)]
    # Doc2Vec用のcsvデータ作成と出力
    dfo = pd.DataFrame()
    dfo["NUMBER"] = df["NUMBER"]
    dfo["SYMBOL"] = df["SYMBOL"]
    dfo["NAME"] = df["NAME"]
    dfo["TOKEN"] = tokens
    outpath = "../atoms_data_for_d2v.csv"
    dfo.to_csv(outpath, encoding="utf-8", index=False)
