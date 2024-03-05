"""陰山さんの出力をもとに、データごとのメンション決定とエンティティ候補生成を行う

    * 陰山さんの出力には、メンションごとの候補エンティティとそのアンカリンク回数が含まれるので、そのデータを用いる
    * 必要なのは、geoNE2anchor.json および filtered_mention2title.json
    * これらは出力ファイルをコピーして data ディレクトリあたりに入れておく
    * エンティティ候補がついたデータには +mention.json とつく
"""

import json
import argparse
from copy import deepcopy
from gensim.models import KeyedVectors
from tqdm import tqdm

class Candidate_Generation():
    def __init__(self, data_name:str, geoNE2mention:str, mention2entity:str) -> None:
        """各データに対し、文書内のメンションとエンティティ候補を生成、保存するクラス。

            * インスタンスを生成して generation() -> save() を呼べばOK

            :param str data_name: 扱うデータの json ファイル名
            :param str geoNE2mention: geoNE と mention の対応辞書データである json ファイル名
            :param str mention2entity: mention からその候補エンティティとアンカリンク回数が保存されている json ファイル名
        """
        self.data_name = data_name

        with open(data_name,'r') as fp:
            self.data = json.load(fp)

        with open(geoNE2mention,'r') as fp:
            self.geoNE2mention = json.load(fp)

        with open(mention2entity,'r') as fp:
            raw_mention2entity = json.load(fp)

        # 日本語Wikipediaエンティティベクトルに存在しないページは取り除く必要がある
        print("Load Word2Vec Data...")
        entity_vector_file = './data/entity_vector.model.bin'
        entity_vector = KeyedVectors.load_word2vec_format( entity_vector_file, binary=True )
        self.mention2entity = deepcopy(raw_mention2entity)

        # ページが無い場合、取り除く処理
        for mention in raw_mention2entity:
            entity_list = raw_mention2entity[mention]
            for entity in entity_list:
                if f'[{entity}]' not in entity_vector:
                    if entity in self.mention2entity[mention]:
                        del self.mention2entity[mention][entity]

        # ページを取り除いた結果、中身が無くなったメンションも取り除く
        delete_mention = []
        for mention,entity_list in self.mention2entity.items():
            if not entity_list: delete_mention.append(mention)
        for mention in delete_mention:
            del self.mention2entity[mention]

    def generation(self) -> None:
        "候補生成の実行"
        for data_type in self.data:
            print('now:',data_type)
            
            for data in tqdm(self.data[data_type]):

                # エンティティ候補は candidate 属性に保存される
                # 各メンションに対し、エンティティ名:リンク回数 のペア
                data['candidate'] = {}
                for geoNE in data['geoNE']:
                    if geoNE not in self.geoNE2mention: continue
                    mention = self.geoNE2mention[geoNE]
                    if mention not in self.mention2entity: continue
                    data['candidate'][mention] = self.mention2entity[mention]

    def save(self) -> None:
        "データを保存する"
        if '+geoNE.json' in self.data_name:
            save_name = self.data_name.replace('+geoNE.json', '+mention.json')
        else:
            save_name = self.data_name.replace('.json', '+mention.json')

        with open(save_name,'w') as fp:
            json.dump(self.data, fp, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--geoNE2mention', default='./data/geoNE2anchor.json')
    parser.add_argument('--mention2entity', default='./data/filtered_mention2title.json')
    parser.add_argument('--target_data', required=True)
    args = parser.parse_args()

    cg = Candidate_Generation(args.target_data, args.geoNE2mention, args.mention2entity)
    cg.generation()
    cg.save()
    print('complete!')