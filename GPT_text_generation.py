from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import datetime
from joblib import Parallel, delayed
import os
import pandas as pd
import argparse
load_dotenv()


def material_2_text(db, i, d, keys, key_names):    
    # check whether the text is already generated
    if os.path.exists(f'{db}_texts/' + db + '_' + str(i) + '.txt'):
        pass  # if the text is already generated, skip
    else:  # Otherwise, generate the text
        dict_mat = {name: d[key] for key, name in zip(keys, key_names)}  # remain nessary info only
        if 'oxide type' in dict_mat.keys():
            if dict_mat['oxide type'] == 'None':
                dict_mat['oxide type'] = 'not oxide'
        
        # Text generation
        response = OpenAI(api_key=os.environ['OPENAI_API_KEY']).chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                    {"role": "system", "content": "You are an excellent materials scientist."},
                    {"role": "user", "content": 'The following dictionary contains the composition and properties of a material stored in the database. \
                     Please write a description of the substance, referring to this information. \
                     Make sure not to omit any item, and include all numerical values, citing their units appropriately. \
                     Feel free to include brief explanations or qualitative meanings for each property. \n' + \
                     str(dict_mat)},
                ]
            ) 
        text = response.choices[0].message.content  
        
        # Save text
        with open(f'{db}_texts/' + db + '_' + str(i) + '.txt', 'w') as textf:
            textf.write(text)
    

def main(db, df):
    # list all targets. 
    if db == 'dft_3d':
        keys = ['formula', 'formation_energy_peratom', 'optb88vdw_bandgap', 'optb88vdw_total_energy', 
                'ehull', 'spg_symbol', 'crys', 'density (g/cm³)', 'volume (Å³)', 'total magnetization (μB/f.u.)',
                'enthalpy per atom (eV/atom)', 'scintillation attenuation length (cm)', 'oxide type', 'stable']
        key_names = ['formula', 'formation energy per atom (eV/atom)', 'band gap (eV)', 'total energy per atom (eV/atom)',
                'energy above hull (eV/atom)', 'space group symbol', 'crystal system', 'density (g/cm³)', 'volume (Å³)', 'total magnetization (μB/f.u.)',
                'enthalpy per atom (eV/atom)', 'scintillation attenuation length (cm)', 'oxide type', 'stable']
    elif db == 'mp_3d_2020':
        keys = ['energy_per_atom', 'volume', 'formation_energy_per_atom', 'enthalpy per atom',
                'pretty_formula', 'e_above_hull', 'is_compatible', 'spacegroup', 'band_gap',
                'density', 'total_magnetization', 'scintillation attenuation length (cm)', 'oxide_type']
        key_names = ['energy per atom (eV/atom)', 'volume (Å³)', 'formation energy per atom (eV/atom)', 'enthalpy per atom (eV/atom)',
                'pretty formula', 'energy above hull (eV/atom)', 'is compatible', 'spacegroup', 'band gap (eV)',
                'density (g/cm³)', 'total magnetization (μB/f.u.)', 'scintillation attenuation length (cm)', 'oxide type']
    elif db == 'aflow2':
        keys = ['band gap (eV)', 'density (g/cm³)',
                'energy above hull (eV/atom)', 'stable', 'volume (Å³)',
                'energy_atom', 'formation energy per atom (eV/atom)',
                'crystal system', 'oxide type', 'elements',
                'enthalpy_atom', 'scintillation_attenuation_length',]
        key_names = ['band gap (eV)', 'density (g/cm³)',
                'energy above hull (eV/atom)', 'stable', 'volume (Å³)',
                'energy per atom (eV/atom)', 'formation energy per atom (eV/atom)',
                'crystal system', 'oxide type', 'elements',
                'enthalpy per atom (eV/atom)', 'scintillation attenuation length (cm)',]
    elif db == 'oqmd':
        keys = ['_oqmd_delta_e', '_oqmd_band_gap', 'formula', '_oqmd_stability', 'scintillation attenuation length (cm)', 
                'Enthalpy per atom (eV/atom)', 'density (g/cm3)', 'crystal system', 'oxide type', 'spacegroup', 'elements', 'stable']
        key_names = ['formation energy per atom (eV/atom)', 'band gap (eV)', 'formula', 'energy above hull (eV/atom)', 'scintillation attenuation length (cm)', 
                'enthalpy per atom (eV/atom)', 'density (g/cm3)', 'crystal system', 'oxide type', 'spacegroup', 'elements', 'stable']

    # Number of texts already generated
    generated = len(os.listdir(f'{db}_texts'))
    print(generated)
    if generated <= 1000:
        num_generated = 0
    else:
        num_generated = generated - 1000 
    # Parallelize text generation
    Parallel(n_jobs=24)(delayed(material_2_text)(db, d[0], d[1], keys, key_names) for d in tqdm(df.iloc[num_generated:].iterrows()))  # TODO 키 이름 변경 후 변경할 것


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text from materials data')
    parser.add_argument('--db', type=str, default='oqmd', help='database name to generate explanation')
    args = parser.parse_args()

    # load data
    df = pd.read_hdf('./data_load/'+f'{args.db}_materials.h5', key=args.db)
    df = df.drop('atoms', axis=1)

    # Make directory to save texts
    os.makedirs(f'{args.db}_texts', exist_ok=True)

    # Text generation
    start = datetime.datetime.now()
    while len(os.listdir(f'{args.db}_texts')) < len(df):  # Repeat until all texts are generated
        main(args.db, df)
    end = datetime.datetime.now() - start
    print(end)

    print('done')
