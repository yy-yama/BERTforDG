"""実験結果を可視化するとき、可視化する対象データを扱うモジュール


"""

import json
from os import path
from utils.argments import args_dg_main

class Data():

    def __init__(self, data_list:list) -> None:
        """実験結果を可視化するとき、可視化する対象データを扱うモジュール

            * 実験結果を保持するためのクラスなので、特にメソッドはありません。
            * データIDをリストで与え、実験結果を読み込みます。

            :param list data_list: データID(str) のリスト
        """
        self.data_list = data_list
        self.attention_list = {}
        self.result = {}
        self.id2word = {}
        self.data_args = {}

        for data in data_list:

            with open(f'./output/attention/{data}.json', 'r') as fp:
                self.attention_list[data] = json.load(fp)

            with open(f'./output/result/{data}.json', 'r') as fp:
                self.result[data] = json.load(fp)

            with open('./model/vocab.txt', 'r') as f:
                self.id2word[data] = {}
                for num,line in enumerate(f):
                    if not line: break
                    self.id2word[data][num] = line.strip()
                # self.id2word[data][7] = '[MEN]'

            with open('summary.txt', 'r') as f:
                target = -1
                args = ''
                for num,line in enumerate(f):
                    if num == target:
                        args = line.rstrip()
                    if line.startswith('Log ID') and data in line:
                        target = num + 2
            
            parser = args_dg_main()
            args = parser.parse_args(args.split()[1:])

            self.data_args[data] = args

            base_data_name = args.data.replace('+entity.json','').replace('.json','')

            # 追加する語彙ファイルが無い場合、リターン
            if not args.EntityEmbedding or 'BERT' in args.EntityEmbedding: continue

            # ConvertedEntityVec 設定の場合、データに関わらず使うのは都道府県のみなのでデータ固定。
            if args.EntityEmbedding == 'ConvertedEntityVec': vocab_file = 'ConvertedEntityVec'
            
            # MentionVec,EntityVecはデータ専用のエンティティ語彙が必要
            else: vocab_file = base_data_name + '_' + args.EntityEmbedding
            
            entity_start = len(self.id2word[data])
            with open(f'./data/entity_vocab/{vocab_file}.txt', 'r') as f:
                for num,line in enumerate(f, entity_start):
                    if not line: break
                    self.id2word[data][num] = line.strip()

    

