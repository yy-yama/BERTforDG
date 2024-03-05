"""文書が分類されたとき、分類の根拠となった Attention の重みをヒートマップを用いて可視化する

    * y.yama 卒論の図 5.3 図 5.4 のような図が出力できる。
    * 1つの出力には1つのデータだけが載るが、複数の実験結果を比較することが可能。
    * ヒートマップ生成に関してさらに詳しくは seaborn.heatmap のドキュメントを参照。
    * https://seaborn.pydata.org/generated/seaborn.heatmap.html
"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import japanize_matplotlib
from utils.visualize.data import Data

class TokenMap():
    def __init__(self, maxlen:int) -> None:
        """1次元のトークン+Attention データを左上から右下へ2次元上にマッピングする。

        :param int maxlen : 何トークン目で折り返すかの指定
        """
        self.maxlen = maxlen
        self.tokenList = [] # トークンを格納する行リスト
        self.attenList = [] # Attention を格納する行リスト
        self.miniTokenList = ['']*self.maxlen # トークンを格納する列リスト
        self.miniAttenList = [float('nan')]*self.maxlen # Attention を格納する列リスト(データが無い場合、NaNにしておくと無視される)
        self.yLabels = [''] # 行のラベル

    def resetLists(self):
        "トークン列を折り返す処理"
        self.tokenList.append(self.miniTokenList)
        self.attenList.append(self.miniAttenList)
        self.yLabels.append('')
        self.miniTokenList = ['']*self.maxlen
        self.miniAttenList = [float('nan')]*self.maxlen

    def writeLists(self, token, atten):
        try:
            # トークン未納箇所を探して格納する
            ind = self.miniTokenList.index('')
        except ValueError:
            # 未納箇所が無い場合、折り返す
            self.resetLists()
            ind = 0

        self.miniTokenList[ind] = token
        self.miniAttenList[ind] = atten

class Cmap_attens():
    def __init__(self, data:Data) -> None:
        """指定されたデータからヒートマップを生成、保存するクラス。

            * 基本的な使い方は、対象データ (複数も可能) をインスタンス生成時に指定して save_png メソッドを呼び出す。
            * 出力画像は、BERTforDG/cmap_attens ディレクトリに保存されます。

            :param Data data: 対象データを格納したData クラスのインスタンス
        """
        self.data = data
        self.load_data = data.data_list[0]
        self.comp_data = data.data_list[1:]

    def make_png_title(self, args, label:str, pred:str) -> str:
        "ヒートマップの上部に実験設定を記述する。その記述内容を生成する。"
        ans = ''

        if not args.EntityEmbedding:
            ans += 'Baseline'
        else:
            ans += args.EntityEmbedding + '-' + args.insert

        ans += f' 正解:{label} 予測:{pred}'
        return ans
    
    def write_tokenMap(self, tm:TokenMap, args, title:str, attention:list, id2word:dict) -> None:
        """与えられたデータをTokenMap上に配置する。
        
            * データの知識付与手法が concat の場合、文字トークンと知識トークンを分ける。
            * データの知識付与手法が infuse の場合、メンションであるトークンにはアスタリスクを付ける。
        """

        tm.miniTokenList[tm.maxlen//2] = title
        tm.miniAttenList[tm.maxlen//2] = 0
        tm.resetLists()
        tm.yLabels[-1] = '投稿テキスト'
        mention_flag = False

        for id, atten in attention:
            token = id2word[id]
            if token == '[SEP]':
                tm.resetLists()
                tm.yLabels[-1] = '居住地情報'
            elif token == '[START]' and args.insert == 'concat':
                tm.resetLists()
                tm.yLabels[-1] = 'エンティティ情報'
            elif args.insert == 'infuse' and token in {'[MENTION]', '[START]', '[END]'}:
                if token == '[MENTION]':
                    mention_flag = True
                elif token == '[START]':
                    mention_flag = False
            else:
                if mention_flag:
                    tm.writeLists('*' + token,atten)
                else:
                    tm.writeLists(token,atten)

        tm.resetLists()

    def save_png(self, num:int) -> None:
        """ヒートマップ画像を保存する。
        """

        MAXLEN = 13

        tm = TokenMap(MAXLEN)

        # データのタイトルを付ける
        label,pred = self.data.result[self.load_data]['Preds'][num]
        title = self.make_png_title(self.data.data_args[self.load_data], label, pred)
        self.write_tokenMap(tm, self.data.data_args[self.load_data], title, self.data.attention_list[self.load_data][num],
                           self.data.id2word[self.load_data] )

        for data in self.comp_data:
            # 比較用のデータがある場合、下に続ける。
            tm.resetLists() # 1行空ける
            label,pred = self.data.result[data]['Preds'][num]
            title = self.make_png_title(self.data.data_args[data], label, pred)
            self.write_tokenMap(tm, self.data.data_args[data], title, self.data.attention_list[data][num],
                            self.data.id2word[data] )

        tm.yLabels.pop(-1)

        fig, ax = plt.subplots(figsize=(12, len(tm.attenList)/2)) 
        sns.heatmap(tm.attenList, vmax=1, vmin=0, center=0.5, cmap='Purples', 
                    annot=tm.tokenList, fmt="", xticklabels=False, yticklabels=tm.yLabels)
        ax.tick_params(axis='y', which='both', left=False, right=False)
        plt.yticks(rotation=0)
        plt.savefig(f'./show/attention_cmap/{self.load_data}-No_{num:04}.png')