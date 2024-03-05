"""ファインチューニング済みBERT でのテストデータ評価用のクラス Eval を提供するモジュール
"""

import torch
import json
from tqdm import tqdm
from utils.BERT.logger import Log
from utils.prefectures import id2pref

class Eval():
    def __init__(self, net) -> None:
        self.net = net
        self.vocab_size = net.config.vocab_size
        self.num_attention_heads = net.config.num_attention_heads
        if hasattr(self.net.bert.embeddings, 'kelayer'):
            self.word_len = self.net.bert.embeddings.word_len
            self.knowledge_len = self.net.bert.embeddings.knowledge_len
        else:
            self.word_len = 0
            self.knowledge_len = 0
        self.AttentionCage = []

        self.log = Log.get_instance()

    def exec(self, test_dl, args):
        # epochの正解数を記録する変数
        epoch_corrects = 0

        test_cnt = 0
        Incorrect = set()
        Correct = set()
        Predict_result = []
        
        for batch in tqdm(test_dl):  # testデータのDataLoader
            # GPUにデータを送る
            inputs = batch['input_ids'].to(self.net.device)  # 文章
            attention_mask = batch['attention_mask'].to(self.net.device) # アテンションマスク
            token_type_ids = batch['token_type_ids'].to(self.net.device)
            labels = batch['label'].to(self.net.device)  # ラベル

            # 順伝搬（forward）計算
            with torch.set_grad_enabled(False):
                # ネットワークに入力
                outputs = self.net(inputs, token_type_ids=token_type_ids, attention_mask=attention_mask,
                    output_attentions=True, output_hidden_states=False, return_dict=True)

                _, preds = torch.max(outputs['logits'], 1)  # ラベルを予測
                epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新
                self.write_attens(batch, outputs['attentions'][-1]) # アテンションを保存

                # 正解/不正解データ、予測都道府県の記録
                cnt = 0
                for x in preds == labels.data:
                    if x: Correct.add(test_cnt)
                    else: Incorrect.add(test_cnt)
                    Predict_result.append([id2pref[int(labels.data[cnt])], id2pref[int(preds[cnt])]])
                    cnt += 1
                    test_cnt += 1

        # 正解率
        epoch_acc = epoch_corrects.double() / len(test_dl.dataset)
        self.log.info('テストデータ{}個での正解率：{:.4f}'.format(len(test_dl.dataset), epoch_acc))
        self.log.end_point(' '.join(args), epoch_acc)

        self.save_attens()
        self.save_results(Correct, Incorrect, Predict_result)


    def write_attens(self, batch, normlized_weights) -> None:
        """アテンション結果を保存する。

        batch: テストデータのミニバッチ
        normlized_weights: アテンションの出力結果
        """

        for index in range(len(batch['input_ids'])):
            # indexの結果を抽出
            sentence = batch['input_ids'][index]  # 文章

            word_pos = torch.nonzero((0 < sentence) & (sentence < self.vocab_size)).squeeze()
            knowledge_pos = torch.nonzero(sentence >= self.vocab_size).squeeze()

            # 12種類のAttentionの平均を求める。最大値で規格化
            attens = torch.zeros_like(normlized_weights[index, 0, 0, :])
            for i in range(self.num_attention_heads):
                attens += normlized_weights[index, i, 0, :]
            attens /= attens.max()

            SEP_cnt = False #2個目のSEP検知用

            word_cnt = 0
            knowledge_cnt = self.word_len

            miniAttensCage = []

            #html += '[BERTのAttentionを可視化_ALL]<br>'
            for num, word in enumerate(sentence):

                # 単語が[SEP] (token ID:3) の場合は文章が終わりなのでbreak
                if int(word.item()) == 3:
                    if SEP_cnt: break
                    SEP_cnt = True

                if num in word_pos:
                    attn = attens[word_cnt]
                    word_cnt += 1
                elif num in knowledge_pos:
                    attn = attens[knowledge_cnt]
                    knowledge_cnt += 1
                else:
                    break

                miniAttensCage.append([word.item(), attn.item()])

            self.AttentionCage.append(miniAttensCage)

    def save_attens(self):
        with open(f"output/attention/{self.log.created_str}.json" , "w") as fp:
            json.dump(self.AttentionCage, fp, ensure_ascii=False)

    def save_results(self, correct, incorrect, preds) -> None:
        with open(f'output/result/{self.log.created_str}.json', "w") as fp:
            json.dump({
                'Correct' : list(correct),
                'Incorrect' : list(incorrect),
                'Preds' : preds
            }, fp, ensure_ascii=False)