"""エンティティ同定が終わったデータをもとに、BERTで使う知識ベクトルとその種類を抽出し、保存する。

    * 出力として、./data/entity_vector および ./data/entity_vocab に吐き出される
"""

import json
import argparse
import numpy as np
from os import path
from gensim.models import KeyedVectors
import torch

class Vector_Extract():
    def __init__(self, data_name:str) -> None:
        """各データに対し、文書内のメンションとエンティティ候補を生成、保存するクラス。

            * インスタンスを生成して save_() を呼べばOK

            :param str data_name: 扱うデータの json ファイル名
        """
        with open(data_name,'r') as fp:
            self.data = json.load(fp)

        print("Load Word2Vec Data...")
        entity_vector_file = './data/entity_vector.model.bin'
        self.entity_vector = KeyedVectors.load_word2vec_format( entity_vector_file, binary=True )

        save_name = path.split(data_name)[-1]
        if '+entity.json' in save_name:
            self.save_name = save_name.replace('+entity.json', '')
        else:
            self.save_name = save_name.replace('.json', '')

    def extract_entity(self) -> list:
        "エンティティ抽出の実行"
        entity_set = set()

        for data_type in self.data:
            for data in self.data[data_type]:
                for ent in data['Entity'].values():
                    # 日本語Wikipediaエンティティベクトルでは、エンティティベクトルのkeyは [エンティティ名] で表される
                    entity_set.add(f'[{ent}]')

        entity_list = list(entity_set)
        entity_list.sort()
        return entity_list

    def save_entity(self) -> None:
        "知識の語彙データと埋め込み表現のテンソルを保存する"
        entity_list = self.extract_entity()
        entity_vector_tensor = torch.tensor([])
        for ent in entity_list:
            # numpy ndarray 形式で保存されている日本語Wikipediaエンティティベクトルを、torch.tensor にする
            entVec = torch.from_numpy( self.entity_vector[ ent ] ).clone().unsqueeze(0)
            # dim 0 で結合 (200次元 × 知識長)
            entity_vector_tensor = torch.cat(( entity_vector_tensor, entVec ), dim=0)
        
        vector_save_name = './data/entity_vector/' + self.save_name + '_EntityVec.pt'
        text_save_name = './data/entity_vocab/' + self.save_name + '_EntityVec.txt'

        torch.save(entity_vector_tensor, vector_save_name)
        with open(text_save_name, 'w') as f:
            f.write('\n'.join(entity_list))

    def extract_MentionVec(self) -> list:
        "MentionVec 抽出の実行"
        mention_dict = {}

        for data_type in self.data:
            for data in self.data[data_type]:
                for men, ents in data['candidate'].items():
                    # 日本語Wikipediaエンティティベクトルでは、エンティティベクトルのkeyは [エンティティ名] で表される
                    mention_dict[f'[{men}]'] = [f'[{e}]' for e in ents]

        MentionVec_list = [(k,v) for k,v in mention_dict.items()]
        MentionVec_list.sort()
        return MentionVec_list

    def save_MentionVec(self) -> None:
        "MentionVec の語彙データと埋め込み表現のテンソルを保存する"
        MentionVec_list = self.extract_MentionVec()
        MentionVec_tensor = torch.tensor([])
        for men, ents in MentionVec_list:
            # エンティティ候補全てのベクトルを平均してMentionVec にする
            vecs = []
            for e in ents:
                vecs.append( self.entity_vector[ e ] )
            MentionVec_np = np.mean( np.stack(vecs) ,0)
            # numpy ndarray 形式のMentionVecを、torch.tensor にする
            MentionVec = torch.from_numpy( MentionVec_np ).clone().unsqueeze(0)
            # dim 0 で結合 (200次元 × 知識長)
            MentionVec_tensor = torch.cat(( MentionVec_tensor, MentionVec ), dim=0)
        
        vector_save_name = './data/entity_vector/' + self.save_name + '_MentionVec.pt'
        text_save_name = './data/entity_vocab/' + self.save_name + '_MentionVec.txt'

        torch.save(MentionVec_tensor, vector_save_name)
        with open(text_save_name, 'w') as f:
            f.write('\n'.join([name for name,_ in  MentionVec_list ] ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_data', required=True)
    args = parser.parse_args()

    ve = Vector_Extract(args.target_data)
    ve.save_entity()
    ve.save_MentionVec()