"""実験結果をLaTeXの表にまとめる。実験結果（精度の一覧）表が作れます。

    * y.yama卒論の表5.4 のような表ができる。
"""

from scipy.stats import binom_test
from io import TextIOWrapper

class Acc_table():
    def __init__(self, method:list, insert:list, baseline:str, result:dict, created:str) -> None:
        """実験結果をLaTeXの表にまとめる。実験結果（精度の一覧）表が作れます。

            * 読み込んだデータを与えてインスタンスを作成する。
            * 出力結果は ./tex/acc_YYYYmmdd-HHMMSS.tex に保存される。
            * LaTeX上では、 \input{acc_YYYYmmdd-HHMMSS.tex} のように記述すると読み込むことができます。
            * y.yama卒論の表5.4 のような表ができる。

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

    def calc_p(self,base:set,targ:set) -> float:
        "P値を計算する"
        successT = len(targ - base)
        successB = len(base - targ)
        total_trials = successT + successB
        expected_success_rate = 0.5
        return binom_test(successT, n=total_trials, p=expected_success_rate)

    def mark(self,base:set,targ:set) -> str:
        "P値に応じて、付与するマークを返す"
        p_value = self.calc_p(base,targ)
        if p_value < 0.01:
            return '^{++}'
        elif p_value < 0.05:
            return '^{+}'
        return ''

    def print(self,file:TextIOWrapper) -> None:
        "表を記述する"

        # tabuler環境のセット
        file.write('\\begin{tabular}{c'+ 'c'*len(self.insert) +'}')

        # ヘッダ行
        file.write('\\hline \n 埋め込み表現 ')
        for i in self.insert:
            file.write(f'& {i} ')
        file.write('\\\\ \n \\hline \n')

        # ベースライン行
        file.write(f'{self.baseline} ')
        for i in self.insert:
            total = len(self.result[i][self.baseline]['Correct']) + len(self.result[i][self.baseline]['Incorrect'])
            acc = 100*len(self.result[i][self.baseline]['Correct']) / total
            file.write(f'& ${acc:.2f}$ ')
        file.write('\\\\ \n')

        # それ以下の行
        for m in self.method:
            if m == self.baseline: continue
            file.write(f'{m} ')
            for i in self.insert:
                total = len(self.result[i][m]['Correct']) + len(self.result[i][m]['Incorrect'])
                acc = 100*len(self.result[i][m]['Correct']) / total

                # 符号検定を行い、有意水準を満たすものにはマークを付ける
                mark = self.mark(self.result[i][self.baseline]['Correct'], self.result[i][m]['Correct'])
                file.write(f'& ${acc:.2f}{mark}$ ')
            file.write('\\\\ \n')
        
        file.write('\\hline \n')

        # tabuler環境の終わり
        file.write('\\end{tabular}')

    def save(self) -> None:
        "表をtexファイルとして保存する"
        with open(f'./show/tex/acc_{self.created}.tex', 'w') as f:
            self.print(f)