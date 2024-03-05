import random
import os
import numpy as np
import torch
from transformers import BertForSequenceClassification
import json
import sys
from utils.argments import args_dg_main
from utils.BERT.logger import Log
from utils.BERT.knowledgeembeddings import KnowledgeEmbeddings
from utils.BERT.dataset import DataSet
from torch.utils.data import DataLoader
from utils.BERT.bert_tokenizer import load_tokenizer
from utils.BERT.train import Train
from utils.BERT.eval import Eval

try:
    log = Log.get_instance()
    parser = args_dg_main()
    args = parser.parse_args()

    if args.EntityEmbedding and args.insert is None:
        log.info('エンティティ埋め込み表現を指定する場合、エンティティの埋め込み手法を [-i] オプションで指定してください。')
        raise Exception
    
    base_data_name = args.data.replace('+entity.json','').replace('.json','')

    cuda = f"cuda:{args.cuda}"

    torch.cuda.init()

    # 乱数のシードを設定
    torch.manual_seed(1234)
    np.random.seed(1234)
    random.seed(1234)

    device = torch.device(cuda if torch.cuda.is_available() else "cpu")

    #BERTの設定
    if not args.load:
        path_model = 'model'
        net = BertForSequenceClassification.from_pretrained(
        path_model, from_tf=True, num_labels=47 #, problem_type="multi_label_classification"
        )
        #problem_type を "multi_label_classification" に設定すると、labelにone_hot vectorを要求される
        arglist = sys.argv
        load_ID = None
        log.set_args(args)
    else:
        load_ID = args.load
        path_model = f'saved_network/{args.load}'
        net = BertForSequenceClassification.from_pretrained(
            path_model, num_labels=47
        )
        with open(f'saved_network/{args.load}/args','r') as f:
            arglist = f.read().split()
        args = parser.parse_args(arglist[1:])
        log.set_args(args)

    net.to(device)

    tokenizer, EntityVec_file = load_tokenizer(args.EntityEmbedding, base_data_name)

    if EntityVec_file:
        entityVecs_path = f'data/entity_vector/{EntityVec_file}.pt'
        entityVecs = torch.load(entityVecs_path)
        entityVecs = entityVecs.to(device)
        net.bert.embeddings = KnowledgeEmbeddings(net.bert.embeddings, entityVecs, args.maxWordLen, args.maxKLen)
    
    net.eval()

    max_length = args.maxWordLen + args.maxKLen

    with open("./data/"+args.data , "r") as fp:
        data = json.load(fp)

    # ツイートデータを読み込んで、データセットに変換
    train_ds = DataSet(tokenizer, data['train'], max_length, args.EntityEmbedding, args.insert, args.entity_pref)
    val_ds = DataSet(tokenizer, data['val'], max_length, args.EntityEmbedding, args.insert, args.entity_pref)
    test_ds = DataSet(tokenizer, data['test'], max_length, args.EntityEmbedding, args.insert, args.entity_pref)

    # DataLoaderを作成します
    batch_size = 16  # BERTでは16、32あたりを使用する

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 辞書オブジェクトにまとめる
    dataloaders_dict = {"train": train_dl, "val": val_dl}

    if not load_ID:
        train = Train(net,args)
        train.exec(dataloaders_dict)
        train.save_network(arglist)
        net_trained = train.return_net()
    else:
        net.bert.embeddings.load_state_dict(torch.load(f'saved_network/{load_ID}/KE.pth'))
        net_trained = net

    net_trained.eval()   # モデルを検証モードに
    net_trained.to(device)  # GPUが使えるならGPUへ送る

    eval = Eval(net_trained)
    eval.exec(test_dl, arglist)

except:
    log.cleanup()