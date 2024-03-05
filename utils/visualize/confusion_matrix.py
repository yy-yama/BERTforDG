"""実験結果の都道府県間の予測誤りや、2つの手法間で誤りが改善された様子をヒートマップとして出力する。

    * y.yama 卒論の図 5.1 図 5.2 のような図が作成できる。
    * ヒートマップ生成に関してさらに詳しくは seaborn.heatmap のドキュメントを参照。
    * https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""
import seaborn as sns
import math
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.visualize.data import Data
from utils.prefectures import regional_prefectures

class Misclassification():
    def __init__(self, data:Data) -> None:
        """実験結果の都道府県間の予測誤りをヒートマップとして可視化,保存するためのクラス

            * 対象データをインスタンス生成時に指定して save_png メソッドを呼び出す。
            * 出力結果は、BERTforDG/confusion_matrix ディレクトリに Miss_XXXX.png として保存されます。

            :param Data data: 対象データを格納したData クラスのインスタンス
        """
        self.data = data
        self.load_data = data.data_list[0]

    def make_pandas(self) -> pd.DataFrame:
        "予測データの誤り部分だけを、行方向が正解/列方向が予測となるように混同行列としてpandas.DataFrameに格納する"
        df = pd.DataFrame(0, index=regional_prefectures, columns=regional_prefectures)
        df.index.name = '正解都道府県'
        df.columns.name = '予測都道府県'

        for label,pred in self.data.result[self.load_data]['Preds']:
            df.at[label, pred] += 1

        # 正解している部分 (対角要素) は 0 にする。
        for p in regional_prefectures: df.at[p, p] = 0

        return df
    
    def save_png(self):
        "画像を保存する"
        df = self.make_pandas()
        vmax = math.ceil( df.max().max()/10 ) * 10
        fig, ax = plt.subplots(figsize=(12, 9)) 
        sns.heatmap(df, square=True, vmax=vmax, vmin=0, center=vmax/2, cmap='gist_heat_r')
        plt.savefig(f'./show/confusion_matrix/Miss_{self.load_data}.png')

class Improvement():
    def __init__(self, data:Data) -> None:
        """2つの手法間で誤りが改善された様子をヒートマップとして可視化,保存するためのクラス

            * 対象データをインスタンス生成時に指定して save_png メソッドを呼び出す。
            * 対象データは2つ指定されていることが前提で、1つ目の結果が2つ目の結果と比較して改善したことを示す。
            * 出力結果は、BERTforDG/confusion_matrix ディレクトリに Comp_XXXX.png として保存されます。

            :param Data data: 対象データを格納したData クラスのインスタンス
        """
        self.data = data
        self.load_data = data.data_list[0]
        self.comp_data = data.data_list[1]

    def make_pandas(self) -> pd.DataFrame:
        "行方向が正解/列方向が予測となるように混同行列としてpandas.DataFrameに格納する"
        df = pd.DataFrame(0, index=regional_prefectures, columns=regional_prefectures)
        df.index.name = '正解都道府県'
        df.columns.name = '予測都道府県'

        # 比較対象データの誤りは、改善の余地であるためプラス
        for label,pred in self.data.result[self.comp_data]['Preds']:
            df.at[label, pred] += 1

        # 優れていることを示したいデータの誤りはマイナスとなる
        for label,pred in self.data.result[self.load_data]['Preds']:
            df.at[label, pred] -= 1

        # 対角要素は消す
        for p in regional_prefectures: df.at[p, p] = 0

        return df
    
    def save_png(self):
        "画像を保存する"
        
        df = self.make_pandas()
        vmax = math.ceil( df.max().max()/10 ) * 10
        vmin = math.floor( df.min().min()/10 ) * 10
        vmax = max(vmax, abs(vmin))
        fig, ax = plt.subplots(figsize=(12, 9)) 
        sns.heatmap(df, square=True, vmax=vmax, vmin=-vmax, center=0, cmap='RdBu')
        plt.savefig(f'./show/confusion_matrix/Comp_{self.load_data}_{self.comp_data}.png')

        

