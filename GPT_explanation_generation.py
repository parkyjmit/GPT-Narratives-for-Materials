from openai import OpenAI
from tqdm import tqdm
import datetime
from joblib import Parallel, delayed
import os
import pandas as pd
import argparse


def text_2_explanation(db, i):  
    # Check if the text is already generated
    if os.path.exists(f'{db}_texts_exp/' + db + '_' + str(i) + '_explanation.txt'):
        pass  # if the text is already generated, skip
    else:  # Otherwise, generate the text
        with open(f'{db}_texts/' + db + '_' + str(i) + '.txt', 'r') as textf:
            text_mat = textf.read()
        
        # Text generation
        response = OpenAI(api_key=os.environ['OPENAI_API_KEY']).chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are an excellent materials scientist."},
                    {"role": "user", "content": "Let's assume that we have a material with the following properties. \
                     Provide possible application areas for this material and explain the rationale behind them. \n" + text_mat},
                ]
            ) 
        text = response.choices[0].message.content  

        # Save text
        with open(f'{db}_texts_exp/' + db + '_' + str(i) + '_explanation.txt', 'w') as textf:
            textf.write(text)
    


def main(db, df):
    # Number of texts already generated
    generated = len(os.listdir(f'{db}_texts_exp'))
    print(generated)
    if generated <= 1000:
        num_generated = 0
    else:
        num_generated = generated - 1000 
    # Text generation
    Parallel(n_jobs=24)(delayed(text_2_explanation)(db, i) for i in tqdm(range(num_generated, len(df))))  # TODO 키 이름 변경 후 변경할 것


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from materials data')
    parser.add_argument('--db', type=str, default='oqmd', help='database name to generate explanation')
    args = parser.parse_args()

    # Load data
    df = pd.read_hdf('./data_load/'+f'{args.db}_materials.h5', key=args.db)
    df = df.drop('atoms', axis=1)

    # Make directories
    os.makedirs(f'{args.db}_texts_exp', exist_ok=True)

    # Text generation
    start = datetime.datetime.now()
    while len(os.listdir(f'{args.db}_texts_exp')) < len(df): 
        main(args.db, df)
    end = datetime.datetime.now() - start
    print(end)

    print('done')
