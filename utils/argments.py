"""BERTforDG 直下から呼び出すプログラムの引数を定義しています。

    * コメントが長くなったのでファイルとして独立させました。
    * 引数の一覧は [-h] オプションを付けることで確認できます。
"""
import argparse

def args_dg_main() -> argparse.ArgumentParser:
    """main.py の引数リストです。
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--cuda', type=int, default=0 ,
                        help="""使用するGPUの番号を指定します。0か1で指定できます。
                        プログラムを動かすとき、GPUメモリを半分以上消費することがあるため、
                        2つの学習を同時に進めたり、他の人と競合する場合に変えることができます。""")
    parser.add_argument('-e', '--epoch', type=int, default=4 ,
                        help="全ての訓練データを何回学習させるか指定します。")
    parser.add_argument('-d', '--data', default='tweet_data+entity.json',
                        help="使用するデータを指定します。")
    parser.add_argument('-e2p', '--entity_pref', default='entity_pref.json',
                        help="エンティティ名から都道府県情報を取得するためのデータを指定します。")
    parser.add_argument('-v', '--EntityEmbedding', default='' , choices=['MentionVec', 'EntityVec', 'ConvertedEntityVec', 'ConvertedBERTVec'],
                        help="""使用するエンティティの埋め込み表現を指定します。
                        指定しない場合、外部知識を用いないベースラインモデルとして動作します。
                        指定する場合、MentionVec, EntityVec, ConvertedEntityVec, ConvertedBERTVec が選択できます。
                        MentionVec, EntityVec を指定する場合は、
                        ./entity_vector にテンソルが入っており、
                        ./entity_vocab にエンティティの一覧が入っている必要があります。
                        両者は語彙数と順序が同じで、[データ名]_[MenionVec/EntityVec] という名前を想定しています。""")
    parser.add_argument('-i', '--insert', choices=['concat', 'infuse'] ,
                        help="""知識埋め込みの取り込み方を指定します。concat か infuse を指定できます。""")
    parser.add_argument('-er', '--EmbStudyRate', type=float, default=5e-6 ,
                        help="""BERTのWord Embeddings層 及び、BERTの全ての層を学習対象にする場合はそれらの学習率です。
                        BERTの最終層付近の学習率や、分類器の学習率と比べて小さな値が良いと思われます。""")
    parser.add_argument('-kr', '--KEStudyRate', type=float, default=1e-3 ,
                        help="""知識埋め込み層の学習率です。大きな値が良いと思われます。""")
    parser.add_argument('-wl', '--maxWordLen', type=int, default=240 ,
                        help="テキストの最大トークン長を指定します。知識埋め込みと合わせて最大長は512まで")
    parser.add_argument('-kl', '--maxKLen', type=int, default=30 ,
                        help="知識埋め込みの最大トークン長を指定します。テキスト埋め込みと合わせて最大長は512まで")
    # parser.add_argument('-d', '--domain', default='' ,
    #                     help="特定ドメインで訓練を行う場合に指定します")
    # parser.add_argument('-dt', '--dataSetType', action='store_false' )
    parser.add_argument('-as', '--AllStudy', action='store_true' ,
                        help="BERTの全ての層を学習対象にする場合に付けるオプションです。")
    parser.add_argument('-l', '--load', default=None ,
                        help="""以前に学習させた設定でもう一度テストデータを評価する場合に指定します。
                        データの指定は、./saved_network のディレクトリ名をそのままIDとして入力し、
                        YYYYmmdd-HHMMSS のかたちで入力します。""")

    return parser

def args_analysis() -> argparse.ArgumentParser:
    """analysis.py の引数リストです。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load', required=True, nargs='+',
                        help="""解析を出力する対象データを指定します。指定必須です。
                        データの指定は、./output/result のファイル名をそのままIDとして入力し、
                        YYYYmmdd-HHMMSS のかたちで入力します。
                        出力するデータによっては複数入力できます。
                        html出力、都道府県間の誤分類ヒートマップは1番目のデータが対象となり、
                        2データ間の差分ヒートマップは1,2番目のデータが対象となります。(2番目がベースライン)""")
    parser.add_argument('-html', '--html', type=int, default=None , nargs=2, 
                        help="""htmlデータとして出力するテストデータの範囲を番号で指定します。
                        0 100 と入力すれば、No.0からNo.99までのデータが出力されます。""")
    parser.add_argument('-na', '--narrow', default=None, choices=['Correct', 'Incorrect'],
                        help="""htmlデータとして出力するテストデータを絞り込みます。
                        Correct / Incorrect と入力し、正解データ、不正解データだけを出力できます。""")
    parser.add_argument('-cmap', '--cmap_attens', default=None, type=int, nargs='+',
                        help="""Attention をカラーマップ画像として出力するテストデータの番号を入力します。
                        複数個指定することも可能です。
                        また、load で複数の対象データを指定した場合、それらを1枚にまとめた画像ができます。""")
    parser.add_argument('-miss', '--miss', action='store_true' ,
                        help="都道府県間の誤分類ヒートマップ(混同行列)を出力する際、立てるフラグです。")
    parser.add_argument('-comp', '--comp', action='store_true' ,
                        help="""2データ間の差分ヒートマップを出力する際、立てるフラグです。
                        load で2個のデータを指定する必要があります。""")
    return parser

def args_table() -> argparse.ArgumentParser:
    """table.py の引数リストです。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--settings', default='table_settings.json' ,
                        help="表を生成するときの設定ファイルです。")
    parser.add_argument('-acc', '--acc', action='store_true' ,
                        help="分類精度の表を出力する際、立てるフラグです。")
    parser.add_argument('-mention', '--mention', action='store_true' ,
                        help="メンション数ごとの結果を出力する際、立てるフラグです。")
    parser.add_argument('-pref', '--pref', action='store_true' ,
                        help="都道府県ごとの結果を出力する際、立てるフラグです。")
    return parser

if __name__ == '__main__':
    pass