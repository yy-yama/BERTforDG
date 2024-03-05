from utils.argments import args_analysis
from utils.visualize.data import Data
from utils.visualize.html_attens import Html_attens
from utils.visualize.cmap_attens import Cmap_attens
from utils.visualize.confusion_matrix import Misclassification, Improvement

parser = args_analysis()
args = parser.parse_args()

data = Data(args.load)

if args.html is not None:
    html = Html_attens(data, args.html, args.narrow)
    html.save_html()
if args.cmap_attens is not None:
    cmap = Cmap_attens(data)
    for num in args.cmap_attens:
        cmap.save_png(num)
if args.miss:
    miss = Misclassification(data)
    miss.save_png()
if args.comp:
    if len(args.load) < 2:
        print('load で2つのデータを指定してください。')
    else:
        imp = Improvement(data)
        imp.save_png()
