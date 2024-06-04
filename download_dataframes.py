from jarvis.db.figshare import data as jdata
import pandas as pd
import numpy as np
from jarvis.core.atoms import Atoms
from jarvis.analysis.structure.spacegroup import Spacegroup3D

def get_crystal_system(row):
    try:
        return Spacegroup3D(Atoms.from_dict(row['atoms'])).crystal_system
    except:
        return 'Unknown'
    

def get_space_group(row):
    try:
        return Atoms.from_dict(row['atoms']).spacegroup()
    except:
        return 'Unknown'
    

def download_dataframes():
    '''
    Download dataframes from JARVIS Figshare.
    '''
    # Download dataframes
    dft_3d = pd.DataFrame(jdata("dft_3d"))
    mp_3d_2020 = pd.DataFrame(jdata("mp_3d_2020"))
    aflow2 = pd.DataFrame(jdata("aflow2"))
    oqmd_3d = pd.DataFrame(jdata("oqmd_3d"))
    oqmd_3d_no_cfid = pd.DataFrame(jdata('oqmd_3d_no_cfid'))

    oqmd_3d['idx'] = oqmd_3d.index
    oqmd_3d_no_cfid['idx'] = oqmd_3d_no_cfid.index + 460046

    oqmd = pd.merge(oqmd_3d, oqmd_3d_no_cfid, on='id', how='outer')
    oqmd['idx'] = oqmd.apply(lambda x: x['idx_x'] if pd.notnull(x['idx_x']) else x['idx_y'], axis=1)
    
    # change name of the column "_oqmd_entry_id" to "id"
    oqmd_3d_no_cfid.rename(columns={'_oqmd_entry_id': 'id'}, inplace=True)
    oqmd = pd.concat([oqmd_3d, oqmd_3d_no_cfid]).groupby('id', as_index=False).first().sort_values('idx')
    # reindex
    oqmd = oqmd.reset_index(inplace=False, drop=True)
    # remove idx column
    oqmd = oqmd.drop(columns=['idx', 'desc'], inplace=False)

    # Add pseudo target for convenience
    dft_3d['pseudo target'] = 0
    mp_3d_2020['pseudo target'] = 0
    aflow2['pseudo target'] = 0
    oqmd['pseudo target'] = 0

    # calculate other properties
    dft_3d['elements'] = dft_3d['atoms'].apply(lambda x: list(set(x['elements'])))
    dft_3d['density (g/cm³)'] = dft_3d['atoms'].apply(lambda x: Atoms.from_dict(x).density)
    dft_3d['space group'] = dft_3d.apply(get_space_group, axis=1)
    dft_3d['formula'] = dft_3d['atoms'].apply(lambda x: Atoms.from_dict(x).composition.reduced_formula)
    dft_3d['crystal system'] =dft_3d.apply(get_crystal_system, axis=1)
    dft_3d['volume (Å³)'] = dft_3d['atoms'].apply(lambda x: Atoms.from_dict(x).volume)

    mp_3d_2020['elements'] = mp_3d_2020['atoms'].apply(lambda x: list(set(x['elements'])))
    mp_3d_2020['density (g/cm³)'] = mp_3d_2020['atoms'].apply(lambda x: Atoms.from_dict(x).density)
    mp_3d_2020['space group'] = mp_3d_2020.apply(get_space_group, axis=1)
    mp_3d_2020['formula'] = mp_3d_2020['atoms'].apply(lambda x: Atoms.from_dict(x).composition.reduced_formula)
    mp_3d_2020['crystal system'] = mp_3d_2020.apply(get_crystal_system, axis=1)
    mp_3d_2020['volume (Å³)'] = mp_3d_2020['atoms'].apply(lambda x: Atoms.from_dict(x).volume)

    aflow2['elements'] = aflow2['atoms'].apply(lambda x: list(set(x['elements'])))
    aflow2['density (g/cm³)'] = aflow2['atoms'].apply(lambda x: Atoms.from_dict(x).density)
    aflow2['space group'] = aflow2.apply(get_space_group, axis=1)
    aflow2['formula'] = aflow2['atoms'].apply(lambda x: Atoms.from_dict(x).composition.reduced_formula)
    aflow2['crystal system'] = aflow2.apply(get_crystal_system, axis=1)
    aflow2['volume (Å³)'] = aflow2['atoms'].apply(lambda x: Atoms.from_dict(x).volume)

    oqmd['elements'] = oqmd['atoms'].apply(lambda x: list(set(x['elements'])))
    oqmd['density (g/cm³)'] = oqmd['atoms'].apply(lambda x: Atoms.from_dict(x).density)
    oqmd['space group'] = oqmd.apply(get_space_group, axis=1)
    oqmd['formula'] = oqmd['atoms'].apply(lambda x: Atoms.from_dict(x).composition.reduced_formula)
    oqmd['crystal system'] = oqmd.apply(get_crystal_system, axis=1)
    oqmd['volume (Å³)'] = oqmd['atoms'].apply(lambda x: Atoms.from_dict(x).volume)
    
    # Save in hdf
    dft_3d.to_hdf('data_load/dft_3d_materials.h5', key='dft_3d')
    mp_3d_2020.to_hdf('data_load/mp_3d_2020_materials.h5', key='mp_3d_2020')
    aflow2.to_hdf('data_load/aflow2_materials.h5', key='aflow2')
    oqmd.to_hdf('data_load/oqmd_materials.h5', key='oqmd_3d')


if __name__ == '__main__':
    download_dataframes()