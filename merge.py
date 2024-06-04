import pandas as pd
import glob
from tqdm import tqdm


def main():
    for db in ['dft_3d', 'mp_3d_2020', 'aflow2', 'oqmd']:
        df = pd.read_hdf('data_load/{db}_materials.h5', key=db)
        texts = []
        for i in tqdm(range(len(df))):
            with open('{db}_texts/{db}_' + str(i) + '.txt', 'r') as f:
                texts.append(f.read())
        df['gpt_text'] = texts
        explanations = []
        for i in tqdm(range(len(df))):
            with open('{db}_texts_exp/{db}_' + str(i) + '_explanation.txt', 'r') as f:
                explanations.append(f.read())
        df['gpt_explanation'] = explanations
        df.to_parquet('{db}_gpt_narratives.parquet')


if __name__ == '__main__':
    main()