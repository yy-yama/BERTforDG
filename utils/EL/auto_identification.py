import json
import argparse
from tqdm import tqdm

class Auto_identification():
    def __init__(self, data_name:str, target_types:list) -> None:
        self.data_name = data_name
        self.target_types = target_types

        with open(data_name,'r') as fp:
            self.data = json.load(fp)

    def exec(self) -> None:
        for data_type in self.data:
            if data_type not in self.target_types: continue
            print('now:',data_type)

            for data in tqdm(self.data[data_type]):
                data['Entity'] = {}
                for mention,ents in data['candidate'].items():
                    # リンク回数の多い順に並び替え、1番上のものと決める。
                    ents = [(v,k) for k,v in ents.items()]
                    ents.sort(reverse=True)
                    data['Entity'][mention] = ents[0][1]

    def save(self) -> None:
        if '+mention.json' in self.data_name:
            save_name = self.data_name.replace('+mention.json', '+entity.json')
        else:
            save_name = self.data_name.replace('.json', '+entity.json')

        with open(save_name,'w') as fp:
            json.dump(self.data, fp, indent=2, ensure_ascii=False)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_data', required=True)
    parser.add_argument('--data_type', nargs='*', choices=['train', 'val', 'test'], required=True)
    args = parser.parse_args()

    ai = Auto_identification(args.target_data, args.data_type)
    ai.exec()
    ai.save()
    print('complete!')