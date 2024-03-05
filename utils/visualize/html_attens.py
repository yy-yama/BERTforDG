"""文書が分類されたとき、分類の根拠となった Attention の重みをHTML形式で可視化する

    * y.yama 卒論では使用せず。主に実験結果の概観や分類がうまくいっているかの確認に使う。
    * 実装は「PyTorchによる発展ディープラーニング」の8章を参考にしています。
    * https://github.com/YutaroOgawa/pytorch_advanced/blob/master/8_nlp_sentiment_bert/8-4_bert_IMDb.ipynb
"""

from utils.visualize.data import Data

class Html_attens():
    
    def __init__(self, data:Data, output_range:tuple = (0,100), narrow:str = "",) -> None:
        """Attention の重みをHTML形式で可視化,保存するためのクラス

            * 基本的な使い方は、対象データ、出力データ範囲をインスタンス生成時に指定して save_html メソッドを呼び出す。
            * 出力結果は、BERTforDG/html ディレクトリに保存されます。
            * デフォルトで出力データ範囲を絞るようになっているのは、全部出力するとデータが重いためです。

            :param Data data: 対象データを格納したData クラスのインスタンス
            :param tuple output_range: 出力データ範囲を(x,y)の形で指定。range(x,y)に変換される
            :param str narrow : オプション。"Correct" or "Incorrect" を指定すると正解/不正解データだけに絞り込み可能。
        """
        self.data = data
        self.output_range = range(output_range[0], output_range[1])
        self.narrow = narrow
        self.load_data = data.data_list[0]
        self.filename = f'./show/html/{self.load_data}_{output_range[0]}-{output_range[1]}{"_" + narrow if narrow else ""}.html'

    # Attentionの可視化に関する設定
    # HTMLを作成する関数を実装
    def highlight(self, word:str, attn:float) -> str:
        "Attentionの値が大きいと文字の背景が濃い赤になるhtmlを出力させる関数"

        html_color = '#%02X%02X%02X' % (
            255, int(255*(1 - attn)), int(255*(1 - attn)))
        return '<span style="background-color: {}"> {}</span>'.format(html_color, word)

    def convert_html(self, data:list, num:int, label:str, pred:str) -> str:
        "HTMLデータを作成する"
        html = 'No.{}<br>正解ラベル：{}<br>推論ラベル：{}<br>'.format(num, label, pred)

        for id, attn in data:
            html += self.highlight(self.data.id2word[self.load_data][id], attn)

        html += "<br><br>"

        return html

    def save_html(self) -> None:
        "HTMLデータを保存する"
        output = ""

        for i in self.output_range:
            if self.narrow and i not in self.data.result[self.load_data][self.narrow]: continue
            label, pred = self.data.result[self.load_data]['Preds'][i]
            output += self.convert_html(self.data.attention_list[self.load_data][i], i, label, pred)

        with open(self.filename, 'w') as f:
            f.write(output)