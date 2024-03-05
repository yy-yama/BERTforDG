"""実験結果をLaTeXの表にまとめる。都道府県ごとのf1スコアの表が作れます。

    * y.yama卒論の表5.9, 5.10 のような表ができる。
"""

import pickle
from io import TextIOWrapper
from collections import Counter
from utils.prefectures import prefectures, regional_prefectures_2D

class Pref_table():
    def __init__(self, method:list, insert:list, baseline:str, result:dict, created:str) -> None:
        """実験結果をLaTeXの表にまとめる。都道府県ごとのf1スコアの表が作れます。

            * 読み込んだデータを与えてインスタンスを作成する。
            * 出力結果は ./tex/pref_insert_YYYYmmdd-HHMMSS_order.tex に保存される。
            * LaTeX上では、 \input{hogehoge.tex} のように記述すると読み込むことができます。
            * y.yama卒論の表5.9,5.10 のような表ができる。

            :param list method: 埋め込み表現の作成手法(str) のリスト
            :param list insert: 埋め込み表現の付与手法(str) のリスト
            :param str baseline: ベースライン手法名
            :param dict result: 各データの実験結果が格納された辞書
            :param str created: 出力結果を保存するための日時情報
        """
        self.method = method
        self.insert = insert
        self.baseline = baseline
        self.result = result
        self.created = created

        # f1 スコアを事前に求めておく
        self.f1 = {}
        self.label_num = {}
        self.data_num = 0

        for i in insert:
            self.f1[i] = {}

            for m in method:
                target = self.result[i][m]['Preds']
                stmp = {p:Counter() for p in prefectures}
                for label,pred in target:
                    stmp[label][pred] += 1
                self.data_num = len(target)

                f1 = {}

                for p in prefectures:
                    labels = 0
                    preds = 0
                    for q in prefectures:
                        labels += stmp[p][q]
                        preds += stmp[q][p]
                    corrects = stmp[p][p]

                    # 精度(precision) = 都道府県 p と予測して正解した文書数 / 都道府県 p と予測した文書数
                    try:
                        precision = corrects / preds
                    except ZeroDivisionError:
                        precision = 1

                    # 再現率(recall) = 都道府県 p と予測して正解した文書数 / 正解が都道府県 p の文書数 
                    try:
                        recall = corrects / labels
                    except ZeroDivisionError:
                        recall = 1

                    try:
                        f1[p] = 100*2*precision*recall / (precision + recall)
                    except ZeroDivisionError:
                        f1[p] = 0
                        
                    self.label_num[p] = labels

                self.f1[i][m] = f1

        # ベースライン性能順にソートした都道府県リストも作成しておく
        self.sorted_prefectures = sorted(prefectures, key=lambda x:self.f1[insert[0]][baseline][x], reverse=True)

    def print_line(self,file:TextIOWrapper, p:str) -> None:
        file.write(p)

        # ベースライン列
        base_f1 = {}
        for i in self.insert:
            file.write(f' & ${self.f1[i][self.baseline][p]:.2f}$')
            base_f1[i] = self.f1[i][self.baseline][p]

        # 提案手法列
        for m in self.method:
            if m == self.baseline: continue
            for i in self.insert:
                mark = '↑' if self.f1[i][m][p] > base_f1[i] else ' '
                file.write(f' & ${self.f1[i][m][p]:.2f}$ & {mark}')

        data_per = self.label_num[p] / self.data_num * 100
        file.write(f' & ${self.label_num[p]}$(${data_per:.2f}$) \\\\ \n')

    def print(self,file:TextIOWrapper, base_order:bool) -> None:
        "表を記述する"

        # tabuler環境のセット
        file.write('\\begin{tabular}{l|' + 'c'*len(self.insert) + '|'+ 'rl'*(len(self.method)-1)*len(self.insert) +'|r}')

        # ヘッダ行
        # ヘッダ行1行目
        file.write('\\hline \n 都道府県 & \\multicolumn{2}{c}{' + self.baseline + '}')
        for m in self.method:
            if m == self.baseline: continue
            file.write('& \n \\multicolumn{4}{c}{' + m + '} ')
        file.write('& 事例数(割合) \\\\ \n')

        # ヘッダ行2行目
        for i in self.insert:
            file.write(f'& {i} ')
        for m in self.method:
            if m == self.baseline: continue
            for i in self.insert:
                file.write('& \n \\multicolumn{2}{c}{' + i + '} ')
        file.write('\\\\ \n \hline \n')

        # それ以下の行
        if base_order:
            for p in self.sorted_prefectures:
                self.print_line(file, p)
            file.write('\\hline \n')
        else:
            for p_cage in regional_prefectures_2D:
                for p in p_cage:
                    self.print_line(file, p)
                file.write('\\hline \n')

        # tabuler環境の終わり
        file.write('\\end{tabular}')

    def save(self) -> None:
        "表をtexファイルとして保存する"

        # 地域区分順の表
        with open(f'./show/tex/pref_{self.created}_regional.tex', 'w') as f:
            self.print(f, False)

        # ベースライン性能順の表
        with open(f'./show/tex/pref_{self.created}_sorted.tex', 'w') as f:
            self.print(f, True)