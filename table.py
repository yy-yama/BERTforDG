import json
from datetime import datetime
from utils.table.acc import Acc_table
from utils.table.mention_num import Mention_num_table
from utils.table.pref import Pref_table
from utils.argments import args_table

parser = args_table()
args = parser.parse_args()

with open(args.settings ,'r') as f:
    settings = json.load(f)

try:
    data_name = settings['data']
    method = settings['method']
    insert = settings['insert']
    baseline = settings['baseline']
    result = {}
    for i in insert:
        result[i] = {}
        for m in method:
            result[i][m] = {}
            with open(f'./output/result/{settings[i][m]}.json', 'r') as fp:
                l = json.load(fp)
                result[i][m]['Correct'] = set(l['Correct'])
                result[i][m]['Incorrect'] = set(l['Incorrect'])
                result[i][m]['Preds'] = l['Preds']

except KeyError:
    print('指定された json ファイルに不備があります。')
    exit()

created = datetime.now().strftime("%Y%m%d-%H%M%S")

if args.acc:
    acc = Acc_table(method,insert,baseline,result,created)
    acc.save()
if args.mention:
    mention = Mention_num_table(method,insert,baseline,result,created,data_name,4)
    mention.save()
if args.pref:
    pref = Pref_table(method,insert,baseline,result,created)
    pref.save()