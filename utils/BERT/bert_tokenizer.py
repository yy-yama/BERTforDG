"""事前学習済みBERTモデルの語彙情報を読み込み、知識語彙を追加する関数の定義。
"""

from transformers import BertJapaneseTokenizer
import os

def load_tokenizer(entity_vocab:str, data_name:str) -> BertJapaneseTokenizer:
    "事前学習済みBERTモデルの語彙情報を読み込み、知識語彙を追加したトークナイザを返す"

    tokenizer_model = 'model'
    tokenizer = BertJapaneseTokenizer.from_pretrained(tokenizer_model)

    # これら3つのトークンはSpecial Tokenとして加える
    tokenizer.add_tokens(['[START]','[END]','[MENTION]'], special_tokens=True)

    # 追加する語彙ファイルが無い場合、リターン
    if not entity_vocab or 'BERT' in entity_vocab: return tokenizer, None

    # ConvertedEntityVec 設定の場合、データに関わらず使うのは都道府県のみなのでデータ固定。
    if entity_vocab == 'ConvertedEntityVec': vocab_file = 'ConvertedEntityVec'
    
    # MentionVec,EntityVecはデータ専用のエンティティ語彙が必要
    else: vocab_file = data_name + '_' + entity_vocab

    # エンティティ語彙はSpecial Tokenとして扱い、bert の tokenizer を通したとき常に1つのトークンとして扱われるようにする
    entity_list = []
    with open(f'data/entity_vocab/{vocab_file}.txt','r') as f:
        for line in f:
            if not line: break
            entity_list.append(line.strip())

    tokenizer.add_tokens(entity_list, special_tokens=True)

    return tokenizer, vocab_file



