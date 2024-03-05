"""BERTのEmbeddingsモジュールと同じ入出力で、知識を扱えるようにしたもの。

    * 実装は「PyTorchによる発展ディープラーニング」の8章やMMBTの実装を参考に我流で作ったものです。
    * MMBT の実装はここ
    * https://github.com/facebookresearch/mmbt
    * pytorch_advanced は BERT の中身を解剖してあるのに近いので参考になります。
"""

import torch
from torch import nn
from copy import deepcopy

# BERT用にLayerNormalization層を定義します。
# 実装の細かな点をTensorFlowに合わせています。
class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """LayerNormalization層です。
        学習済みモデルをそのままロードするため、学習済みモデルの変数名に変えています。
        オリジナルのGitHubの実装から変数名を変えています。
        weight→gamma、bias→beta
        """
        super(BertLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))  # weightのこと
        self.beta = nn.Parameter(torch.zeros(hidden_size))  # biasのこと
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class KnowledgeEmbeddings(nn.Module):
    def __init__(self, _bert_embeddings, _entityVec=None, _word_len=256, _knowledge_len=0):
        """BERTのEmbeddingsモジュールを付け替えて知識を扱えるようにします。
        単語トークンとエンティティトークンが混ざった状態で入力されるので、
        それらを仕分け、エンティティトークンはエンティティベクトルに変換した後
        変換行列を通してから単語埋め込みと結合して下層へ流す。

        _bert_embeddings: BERTのもともとのEmbeddingsモジュール
        _entityVec: エンティティベクトルのテンソル [エンティティ数, エンティティベクトルの次元]
        _word_len: 単語トークンだけの最大長
        _knowledge_len: エンティティトークンだけの最大長
        """
        super(KnowledgeEmbeddings, self).__init__()
        self.device = _bert_embeddings.word_embeddings.weight.device
        self.word_len = _word_len
        self.knowledge_len = _knowledge_len

        # word/position/token_type embeddingsに関してはbertのものをそのまま利用
        bert_embeddings = deepcopy(_bert_embeddings)
        self.word_embeddings = bert_embeddings.word_embeddings.to(self.device)
        self.position_embeddings = bert_embeddings.position_embeddings.to(self.device)
        self.token_type_embeddings = bert_embeddings.token_type_embeddings.to(self.device)
        self.word_LayerNorm = bert_embeddings.LayerNorm.to(self.device)
        self.word_dropout = bert_embeddings.dropout.to(self.device)
        self.hidden_size = self.word_embeddings.weight.size(-1)
        self.vocab_size = self.word_embeddings.weight.size(0)

        # エンティティベクトル（既知）、知識埋め込み用変換行列を作成、初期化
        self.entityVec = _entityVec
        self.entityVec_features = self.entityVec.size(-1)
        self.kelayer = nn.Linear(in_features=self.entityVec_features, out_features=self.hidden_size).to(self.device)

        nn.init.normal_(self.kelayer.weight, std=0.02)
        nn.init.normal_(self.kelayer.bias, 0)

        # 知識埋め込み用正規化層とドロップアウトを定義
        self.knowledge_LayerNorm = BertLayerNorm(self.hidden_size).to(self.device)
        self.knowledge_dropout = nn.Dropout(0.1).to(self.device)

    def forward(self, input_ids, token_type_ids, *input, **kwargs) -> torch.tensor:
        '''
        input_ids： [batch_size, sequence_length]の文章の単語IDの羅列
        token_type_ids： [batch_size, sequence_length]の、各単語が1文目なのか、2文目なのかを示すid
        '''

        seq_length = input_ids.size(1)  # 文章の長さ

        # torchで range(seq_length) をやっています
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)

        # input_ids を切り分けて word と knowledge に分けるので、空の状態で用意します
        word_ids = torch.tensor([], dtype=torch.long, device=input_ids.device)
        word_token_type_embeddings = torch.tensor([], device=input_ids.device)
        word_position_embeddings = torch.tensor([], device=input_ids.device)

        # 知識トークンをエンティティベクトルに変換してここに入れる
        input_entityVecs = torch.tensor([], device=input_ids.device)
        knowledge_token_type_embeddings = torch.tensor([], device=input_ids.device)
        knowledge_position_embeddings = torch.tensor([], device=input_ids.device)

        # 知識が入っていない部分を勾配計算に入れないためのマスク
        knowledge_mask = torch.tensor([], device=input_ids.device)      
        
        for i in range(len(input_ids)):
            # ここ (nn.Module の forward 内) でfor文を使っても良いのかは不明
            # なんかGPU上での並列計算とかに悪影響が出そう。改善の余地あり。

            input_id = input_ids[i]
            token_type_id = token_type_ids[i]

            # 文書部分と知識部分をそれぞれ抜き出す。
            word_id_position = torch.nonzero((0 < input_id) & (input_id < self.vocab_size)).squeeze()
            knowledge_id_position = torch.nonzero(input_id >= self.vocab_size).squeeze()

            # 文書埋め込み部分の作成
            word_id = torch.zeros(self.word_len, dtype=torch.long, device=input_ids.device)
            word_id[:len(word_id_position)] = input_id[word_id_position]
            word_ids = torch.cat([word_ids, word_id.unsqueeze(0)])

            word_token_type_id = torch.ones(self.word_len, dtype=torch.long, device=input_ids.device)
            word_token_type_id[:len(word_id_position)] = token_type_id[word_id_position]
            word_token_type_embedding_tmp = self.token_type_embeddings(word_token_type_id)
            word_token_type_embeddings = torch.cat(
                [word_token_type_embeddings, word_token_type_embedding_tmp.unsqueeze(0)])

            word_position_id = torch.arange(self.word_len, dtype=torch.long, device=input_ids.device)
            word_position_id[:len(word_id_position)] = position_ids[word_id_position]
            word_position_embedding_tmp = self.position_embeddings(word_position_id)
            word_position_embeddings = torch.cat(
                [word_position_embeddings, word_position_embedding_tmp.unsqueeze(0)])

            # 使用するエンティティベクトルの抜き出し
            use_entityVecs = torch.zeros(self.knowledge_len, self.entityVec_features, dtype=torch.float, device=input_ids.device)
            knowledge_id = input_id[knowledge_id_position] - self.vocab_size
            knowledge_token_type_id = torch.zeros(self.knowledge_len, dtype=torch.long, device=input_ids.device)
            knowledge_position_id = torch.zeros(self.knowledge_len, dtype=torch.long, device=input_ids.device)

            # 知識がある場合とない場合に分けて処理を行っている
            if knowledge_id_position.size():
                kn = len(knowledge_id_position)
                use_entityVecs[:kn] = self.entityVec[knowledge_id] # size[MAX_ENTS_NUM,self.entityVec_features] のハズ
                knowledge_mask = torch.cat(
                    [knowledge_mask, torch.cat(
                        [torch.ones(kn, self.hidden_size, device=input_ids.device),
                         torch.zeros(self.knowledge_len - kn, self.hidden_size, device=input_ids.device)]
                    ).unsqueeze(0)]
                )
                knowledge_token_type_id[:kn] = token_type_id[knowledge_id_position]
                knowledge_token_type_embedding_tmp = self.token_type_embeddings(knowledge_token_type_id)
                knowledge_position_id[:kn] = position_ids[knowledge_id_position]
                knowledge_position_embedding_tmp = self.position_embeddings(knowledge_position_id)

            else:
                knowledge_mask = torch.cat(
                    [knowledge_mask, torch.zeros(1, self.knowledge_len, self.hidden_size, device=input_ids.device)]
                )
                knowledge_token_type_embedding_tmp = torch.zeros(self.knowledge_len, self.hidden_size, device=input_ids.device)
                knowledge_position_embedding_tmp = torch.zeros(self.knowledge_len, self.hidden_size, device=input_ids.device)

            input_entityVecs = torch.cat([input_entityVecs, use_entityVecs.unsqueeze(0)])
            knowledge_token_type_embeddings = torch.cat(
                [knowledge_token_type_embeddings, knowledge_token_type_embedding_tmp.unsqueeze(0)])
            knowledge_position_embeddings = torch.cat(
                [knowledge_position_embeddings, knowledge_position_embedding_tmp.unsqueeze(0)])

        # size[32, MAX_ENTS_NUM, hidden_size] のハズ
        knowledge_embeddings = self.kelayer(input_entityVecs) + knowledge_token_type_embeddings + knowledge_position_embeddings
        knowledge_embeddings = torch.mul(knowledge_embeddings, knowledge_mask) # 知識が埋まっていない部分は0に補正する
        knowledge_embeddings = self.knowledge_LayerNorm(knowledge_embeddings)
        knowledge_embeddings = self.knowledge_dropout(knowledge_embeddings)

        word_id_embeddings = self.word_embeddings(word_ids)
        word_embeddings = word_id_embeddings + word_token_type_embeddings + word_position_embeddings
        word_embeddings = self.word_LayerNorm(word_embeddings)
        word_embeddings = self.word_dropout(word_embeddings)

        embeddings = torch.cat([word_embeddings, knowledge_embeddings], 1) # size[batch_size,max_length,hidden_size] のハズ

        return embeddings