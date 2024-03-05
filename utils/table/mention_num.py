"""実験結果をLaTeXの表にまとめる。メンション数ごとの実験結果（精度の一覧）表が作れます。

    * y.yama卒論の表5.5, 5.6 のような表ができる。
"""

import json
from io import TextIOWrapper
from collections import defaultdict

class Mention_num_table():
    def __init__(self, method:list, insert:list, baseline:str, result:dict, created:str, data:str, men_max:int) -> None:
        """実験結果をLaTeXの表にまとめる。メンション数ごとの実験結果（精度の一覧）表が作れます。

            * 読み込んだデータを与えてインスタンスを作成する。
            * メンション数をカウントするため、実験に使用したデータの読み込みが追加で必要。
            * 出力結果は ./tex/mention_insert_YYYYmmdd-HHMMSS.tex に保存される。
            * LaTeX上では、 \input{acc_YYYYmmdd-HHMMSS.tex} のように記述すると読み込むことができます。
            * y.yama卒論の表5.5,5.6 のような表ができる。

            :param list method: 埋め込み表現の作成手法(str) のリスト
            :param list insert: 埋め込み表現の付与手法(str) のリスト
            :param str baseline: ベースライン手法名
            :param dict result: 各データの実験結果が格納された辞書
            :param str created: 出力結果を保存するための日時情報
            :param str data: 実験に使用したデータ
            :param str men_max: 表記メンション数の最大値(X以上と表記される)
        """
        self.method = method
        self.insert = insert
        self.baseline = baseline
        self.result = result
        self.created = created

        # データのメンション数を数える
        with open(f'./data/{data}','r') as fp:
            tweet_data = json.load(fp)
        
        self.mention_counter = defaultdict(set)
        for num,d in enumerate(tweet_data['test']):
            men_num = len(d['Entity'])
            men_num = min(men_num, men_max)
            self.mention_counter[ men_num ].add(num)

        self.men_max = men_max
        self.data_num = len(tweet_data['test'])

    def print(self,file:TextIOWrapper, insert:str) -> None:
        "表を記述する"

        # tabuler環境のセット
        file.write('\\begin{tabular}{r|c|' + 'rl'*(len(self.method)-1) +'|r}')

        # ヘッダ行
        file.write(f'\\hline \n メンション数 & {self.baseline}')
        for m in self.method:
            if m == self.baseline: continue
            file.write('& \n \\multicolumn{2}{c}{' + m + '} ')
        file.write('& 事例数(割合) \\\\ \n \\hline \n')

        # それ以下の行
        for i in range(self.men_max + 1):
            file.write(f'{i}')
            if i == self.men_max: file.write('以上')

            base_correct = self.result[insert][self.baseline]['Correct'] & self.mention_counter[i]
            base_acc = len(base_correct)/len(self.mention_counter[i])*100
            file.write(f' & ${base_acc:.2f}$')

            for m in self.method:
                if m == self.baseline: continue
                correct = self.result[insert][m]['Correct'] & self.mention_counter[i]
                acc = len(correct)/len(self.mention_counter[i])*100
                mark = '↑' if acc > base_acc else ' '
                file.write(f' & ${acc:.2f}$ & {mark}')

            data_per = len(self.mention_counter[i]) / self.data_num * 100
            file.write(f' & ${len(self.mention_counter[i])}$(${data_per:.2f}$) \\\\ \n')
        
        file.write('\\hline \n')

        # tabuler環境の終わり
        file.write('\\end{tabular}')

    def save(self) -> None:
        "表をtexファイルとして保存する"

        for i in self.insert:
            with open(f'./show/tex/mention_{self.created}_{i}.tex', 'w') as f:
                self.print(f,i)