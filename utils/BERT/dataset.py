"""BERTでの学習・推論時にデータをミニバッチとして取り出せる DataSet を提供するモジュール。

    * 文書に付随したエンティティ情報を文書内に埋め込む機能を備えている。
"""

import torch
import json
from utils.prefectures import pref2id

class DataSet:
    def __init__(self, _tokenizer, _data:list, _length:int, _entity_embedding:str, _insert:str, entity_pref:str):
        '''BERTでTwitterデータセットの分類を行うためのデータセットを作成する。

        _tokenizer: BERTのトークナイザ。事前に語彙の追加やSpecial Tokenの設定をすませる。
        _data: リスト形式でツイートテキスト、居住地情報、ラベル等が含まれたデータを渡す。
        _length: トークンの最大長。
        '''
        self.text = []
        self.location = []
        self.label = []
        self.info = []

        with open(f'./data/{entity_pref}', 'r') as f:
            ent2pref = json.load(f)

        if not _entity_embedding:
            pickup = None
            self.use_info = None
        elif _entity_embedding.startswith('Convert'):
            pickup = 'Convert'
            self.use_info = _entity_embedding[:-3]
        else:
            pickup = _entity_embedding[:-3]
            self.use_info = _entity_embedding[:-3]
        # if _entity_embedding.endswith('EntityVec'):
        #     pickup = 'Entity'
        #     self.use_info = 'Entity'
        # elif _entity_embedding.endswith('MentionVec'):
        #     pickup = 'Mention'
        #     self.use_info = 'Mention'

        for d in _data:
            self.text.append(d['text'])
            self.location.append(d['location'])
            self.label.append( pref2id[d['label']] )
            if pickup:
                if pickup == 'Entity':
                    self.info.append( d[pickup] )
                elif pickup == 'Mention':
                    men2men = {}
                    for men in d['candidate']:
                        men2men[men] = men
                    self.info.append(men2men)
                else:
                    men2pref = {}
                    for men,ent in d['Entity'].items():
                        men2pref[men] = ent2pref[ent]
                    self.info.append(men2pref)

        self.tokenizer = _tokenizer
        self.length = _length
        self.insert = _insert

    def __len__(self) -> int:
        return len(self.text) # データ数を返す

    def __getitem__(self, index) -> dict:
        """index番目の入出力をdict形式で返す。
        """

        text = self.text[index]
        location = self.location[index]

        if not self.use_info:
            # ベースラインなので、何もしない
            pass
            
        else:
            # 文書内に特殊トークンと知識トークンを挿入する
            no_brackets = 'BERT' in self.use_info

            if self.insert == 'concat':
                headlist, taillist = [], []         
                for k,v in self.info[index].items():
                    text_find = text.find(k)
                    loc_find = location.find(k)

                    if text_find > -1:
                        if no_brackets:
                            headlist.append((text_find, f'{v}' ))
                        else:
                            headlist.append((text_find, f'[{v}]' ))

                    if loc_find > -1:
                        if no_brackets:
                            taillist.append((loc_find, f'{v}' ))
                        else:
                            taillist.append((loc_find, f'[{v}]' ))

                headlist.sort()
                taillist.sort()

                text += ''.join(['[START]'] + [b for a,b in headlist])
                location += ''.join(['[START]'] + [b for a,b in taillist])

            elif self.insert == 'infuse':
                replace_text = text
                replace_loc = location
                replace_dict = {}
                for idx, (k,v) in enumerate( self.info[index].items() ):
                    replace_text = replace_text.replace(k, f'[unused{idx}]')
                    replace_loc = replace_loc.replace(k, f'[unused{idx}]')

                    if no_brackets:
                        replace_dict[f'[unused{idx}]'] = f'[MENTION]{k}[START]{v}[END]'
                    else:
                        replace_dict[f'[unused{idx}]'] = f'[MENTION]{k}[START][{v}][END]'

                for k,v in replace_dict.items():
                    replace_text = replace_text.replace(k,v)
                    replace_loc = replace_loc.replace(k,v)

                text = replace_text
                location = replace_loc
        
        # tokenizer にはテキストを2つ与えることにより自動的に1文目、2文目として扱われる
        # truncation は max_length を超えたときに切り詰めるかどうか。
        # padding は max_length の長さまで [PAD] で埋める設定。
        tok = self.tokenizer(text, location, max_length=self.length, truncation=True, padding="max_length")
        
        # dict 形式でリターンする
        return {
            'input_ids' : torch.tensor(tok['input_ids'], dtype=torch.long),
            'token_type_ids' : torch.tensor(tok['token_type_ids'], dtype=torch.long),
            'attention_mask' : torch.tensor(tok['attention_mask'], dtype=torch.long),
            'label' : self.label[index]
        }