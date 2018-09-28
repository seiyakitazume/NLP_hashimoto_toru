# -------------------
# Module import
# -------------------
import re
import os
import time
import tqdm
import codecs
import pandas as pd
from libs.wakati import Wakati

# -------------------
# Class definition
# -------------------
class TokenData:
    """わかちがきした結果を出力するためのクラス."""

    def __init__(self, path_dir_data):
        """コンストラクタ."""
        self.path_dir_data = path_dir_data

    def _get_filenames(self):
        """path_dir_data 以下の.txtファイル名リストを返す."""
        filenames = os.listdir(self.path_dir_data)
        filenames = [filename for filename in filenames if ".txt" in filename]
        return filenames

    def _get_filepaths(self, filenames):
        """path_dir_data 以下の.txtファイルパスを返す."""
        filepaths = [os.path.join(self.path_dir_data, filename) for filename in filenames]
        return filepaths

    def _get_text(self, filepath):
        """テキストファイルからテキストを抽出して返す."""
        f = codecs.open(filepath, "r", "cp932", "ignore")
        text = f.read()
        f.close()
        return text

    def get_df_tokenized(self):
        """テキストをわかちがきした結果をDataFrameで返す."""
        filenames = self._get_filenames()
        filepaths = self._get_filepaths(filenames)
        texts = [self._get_text(filepath) for filepath in tqdm.tqdm(filepaths)]
        tokenized_texts = [" ".join(Wakati(text).tokenize()) for text in tqdm.tqdm(texts)]
        df_tokenized = pd.DataFrame()
        df_tokenized["filename"] = filenames
        df_tokenized["token"] = tokenized_texts
        return df_tokenized

    def do(self, path_out="../tokenized_data.csv"):
        """テキストをわかちがきした結果をcsvファイルで出力する."""
        self.df_tokenized = self.get_df_tokenized()
        self.df_tokenized.to_csv(path_out, encoding="utf8", index=False)

# -------------------
# Main processing
# -------------------
if __name__=="__main__":
    path_dir_data = "../data/"
    td = TokenData(path_dir_data)
    td.do()
    print(td.df_tokenized.head())
