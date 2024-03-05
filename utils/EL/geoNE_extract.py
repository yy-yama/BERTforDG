"""GiNZA を使ってメンションの元となる geoNE を抽出し、保存するプログラム

    * 抽出対象の固有表現クラスは、デフォルトでは target_ne_label.txt に書かれているものです (陰山さんの設定)
    * 出力はテキスト形式の一覧と、対象データに属性 "geoNE" を加えた json データ
    * 入力: hogehoge.json 出力: hogehoge_geoNE.txt と hogehoge+geoNE.json
"""

import spacy
from tqdm import tqdm
import json
import argparse

class GeoNE_Extract():
    def __init__(self, geoNE_label_data:str, target_data:str) -> None:
        """geoNE を抽出、保存するクラス。

            * インスタンスを生成して save() を呼べばOK

            :param str geoNE_label_data: 抽出対象の固有表現クラスが一覧で書かれたテキストのファイル名
            :param str target_data: 扱うデータの json ファイル名
        """
        self.ne_label_set = set()
        with open(geoNE_label_data, 'r') as file:
            # ファイルの各行を処理
            for line in file:
                # 改行文字を削除してからセットに追加
                self.ne_label_set.add(line.strip())

        self.target_name = target_data
        with open(target_data,'r') as fp:
            self.targ_data = json.load(fp)

        self.ginza = spacy.load('ja_ginza_electra')

    def extract(self) -> list:
        "geoNE を抽出し、リストとして返す。データにも挿入する。"
        ans_set = set()

        for data_type in self.targ_data:
            print('now:',data_type)

            for data in tqdm(self.targ_data[data_type]):
                geoNE = set()

                for key in ['text', 'location']:
                    doc = self.ginza(data[key])
                    for ent in doc.ents:
                        if ent.label_ in self.ne_label_set:
                            geoNE.add(ent.text)

                # json では set 型が扱えないのでリスト形式に変換
                data['geoNE'] = list(geoNE)
                ans_set |= geoNE

        ans_list = list(ans_set)
        ans_list.sort()

        return ans_list

    def save(self) -> None:
        "抽出と保存の実行"
        geoNE_list = self.extract()
        save_text = self.target_name.replace('.json', '_geoNE.txt')
        save_name = self.target_name.replace('.json', '+geoNE.json')

        with open(save_text, 'w') as f:
            for geoNE in geoNE_list:
                f.write(geoNE + '\n')

        with open(save_name,'w') as fp:
            json.dump(self.targ_data, fp, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ne_label', default='./data/target_ne_label.txt')
    parser.add_argument('--target_data', required=True)
    args = parser.parse_args()

    geoNE = GeoNE_Extract(args.ne_label, args.target_data)
    geoNE.save()
    print('complete!')

