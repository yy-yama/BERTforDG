"""BERTのファインチューニング用のクラス Train を提供するモジュール
"""

import torch
import torch.optim as optim
import time
from utils.BERT.logger import Log
import os

class Train():
    def __init__(self, net, config) -> None:
        self.log = Log.get_instance()
        self.net = net
        self.num_epoch = config.epoch
        use_kelayer = os.path.isfile(f'data/entity_vector/{config.EntityEmbedding}.pt')

        # 1. まず全部を、勾配計算Falseにしてしまう
        for name, param in self.net.named_parameters():
            self.log.debug(name)
            param.requires_grad = config.AllStudy

        # 2. 最後のBertLayerモジュールを勾配計算ありに変更
        for name, param in self.net.bert.encoder.layer[-4:].named_parameters():
            param.requires_grad = True

        if use_kelayer:
            # 2.1 知識埋め込み層を勾配計算ありに変更
            for name, param in self.net.bert.embeddings.kelayer.named_parameters():
                param.requires_grad = True

            # 2.2 知識埋め込み層の正規化層を勾配計算ありに変更
            for name, param in self.net.bert.embeddings.knowledge_LayerNorm.named_parameters():
                param.requires_grad = True

        # 3. 識別器を勾配計算ありに変更
        for name, param in self.net.classifier.named_parameters():
            param.requires_grad = True

        # 最適化手法の設定
        # BERTの元の部分はファインチューニング

        optim_list = [
            {'params': self.net.bert.embeddings.word_embeddings.parameters(), 'lr': config.EmbStudyRate},
            {'params': self.net.bert.encoder.layer[:-4].parameters(), 'lr': config.EmbStudyRate},
            {'params': self.net.bert.encoder.layer[-4].parameters(), 'lr': max(5e-6, config.EmbStudyRate)}, 
            {'params': self.net.bert.encoder.layer[-3].parameters(), 'lr': max(1e-5, config.EmbStudyRate)},
            {'params': self.net.bert.encoder.layer[-2].parameters(), 'lr': max(2e-5, config.EmbStudyRate)},
            {'params': self.net.bert.encoder.layer[-1].parameters(), 'lr': max(5e-5, config.EmbStudyRate)},
            {'params': self.net.classifier.parameters(), 'lr': 5e-5}
        ]

        if use_kelayer:
            optim_list.extend([ 
                {'params': self.net.bert.embeddings.word_LayerNorm.parameters(), 'lr': config.EmbStudyRate},
                {'params': self.net.bert.embeddings.kelayer.parameters(), 'lr': config.KEStudyRate},
                {'params': self.net.bert.embeddings.knowledge_LayerNorm.parameters(), 'lr': config.KEStudyRate}
            ])
        else:
            optim_list.extend([ 
                {'params': self.net.bert.embeddings.LayerNorm.parameters(), 'lr': config.EmbStudyRate}
            ])

        self.log.debug(optim_list)
        
        self.optimizer = optim.Adam(optim_list, betas=(0.9, 0.999))

    def exec(self, dataloaders_dict:dict) -> None:

        self.log.info(f"使用デバイス : {self.net.device}")
        self.log.info('-----start-------')

        # ネットワークがある程度固定であれば、高速化させる
        torch.backends.cudnn.benchmark = True

        # ミニバッチのサイズ
        batch_size = dataloaders_dict["train"].batch_size

        # epochのループ
        for epoch in range(self.num_epoch):
            # epochごとの訓練と検証のループ
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.net.train()  # モデルを訓練モードに
                else:
                    self.net.eval()   # モデルを検証モードに

                epoch_loss = 0.0  # epochの損失和
                epoch_corrects = 0  # epochの正解数
                iteration = 1

                # 開始時刻を保存
                t_epoch_start = time.time()
                t_iter_start = time.time()

                # データローダーからミニバッチを取り出すループ
                for batch in (dataloaders_dict[phase]):
                    # batchは辞書型変数

                    # GPUにデータを送る
                    inputs = batch['input_ids'].to(self.net.device)  # 文章
                    attention_mask = batch['attention_mask'].to(self.net.device) # アテンションマスク
                    token_type_ids = batch['token_type_ids'].to(self.net.device)
                    labels = batch['label'].to(self.net.device)  # ラベル

                    # optimizerを初期化
                    self.optimizer.zero_grad()

                    # 順伝搬（forward）計算
                    with torch.set_grad_enabled(phase == 'train'):
                        # ネットワークに入力
                        outputs = self.net(input_ids=inputs, labels=labels, token_type_ids=token_type_ids, attention_mask=attention_mask,
                            output_attentions=False, output_hidden_states=False, return_dict=True)

                        _, preds = torch.max(outputs['logits'], 1)  # ラベルを予測

                        # 訓練時の処理
                        if phase == 'train':
                            # 誤差を逆伝播
                            outputs['loss'].backward()
                            self.optimizer.step()

                            if (iteration % 100 == 0):  # 100iterに1度、lossを表示
                                t_iter_finish = time.time()
                                duration = t_iter_finish - t_iter_start
                                acc = (torch.sum(preds == labels.data)
                                    ).double()/batch_size
                                self.log.info('イテレーション {} || Loss: {:.4f} || 100iter: {:.4f} sec. || 本イテレーションの正解率 : {}'.format(
                                    iteration, outputs['loss'].item(), duration, acc))
                                t_iter_start = time.time()

                        iteration += 1

                        # 損失と正解数の合計を更新
                        epoch_loss += outputs['loss'].item() * batch_size
                        epoch_corrects += torch.sum(preds == labels.data)

                # epochごとのlossと正解率
                epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
                epoch_acc = epoch_corrects.double(
                ) / len(dataloaders_dict[phase].dataset)

                self.log.info('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, self.num_epoch,
                                                                            phase, epoch_loss, epoch_acc))


    def return_net(self):
        return self.net

    def save_network(self, args):
        self.net.save_pretrained(f'output/saved_network/{self.log.created_str}')
        torch.save(self.net.bert.embeddings.state_dict(), f'output/saved_network/{self.log.created_str}/KE.pth')
        with open(f'output/saved_network/{self.log.created_str}/args','w') as f:
            f.write(' '.join(args))
