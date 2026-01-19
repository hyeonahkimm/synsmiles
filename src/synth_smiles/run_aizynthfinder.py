import pickle
import sqlite3
import pandas as pd
import os
from tqdm import tqdm
from aizynthfinder.aizynthfinder import AiZynthFinder

SMILIES_RST_PATH = '/network/scratch/k/kimh/projects/RxnFlow/src/synth_smiles/outputs'
SFN_RST_PATH = '/network/scratch/k/kimh/projects/sfn_chemp/exp/logs'


def get_data(run_name, task = 'SEH', phase='final', unique=True):

    if phase == 'final':
        df = pd.read_csv(f"{SMILIES_RST_PATH}/{task}/{run_name}_final.csv")
        if unique:
            df = df.drop_duplicates(subset=['smiles'])
        print('after droping duplications:', len(df.drop_duplicates(subset=['smiles'])))
        return df['smiles'].tolist()
    elif phase == 'diverse_topk':
        df = pd.read_csv(f"{SMILIES_RST_PATH}/{task}/{run_name}_final_diverse_topk.csv")
        return df['smiles'].tolist()
    else:
        with open(f"{SMILIES_RST_PATH}/{task}/{run_name}_positive_replay.pkl", "rb") as f:
            rb = pickle.load(f)
        return [t.smiles for t in rb]


def get_data_from_db(run_name, phase='final', unique=True):
    db_path = os.path.abspath( os.path.join(SFN_RST_PATH, run_name, phase, 'generated_objs_0.db'))
    df = pd.read_sql_query("SELECT * FROM results", sqlite3.connect(db_path))
    df = df.sample(1000)

    print('befor droping duplications:', len(df))

    if unique:
        df = df.drop_duplicates(subset=['smi'])  # smiles are canonicalized
        # df = df.sort_values(by='r', ascending=False)
        # df = df[df['fr_0'] > 0.0]

    print('after droping duplications:', len(df.drop_duplicates(subset=['smi'])))

    return df['smi'].tolist()


if __name__ == "__main__":
    # task = 'SEH'
    # sfn_run_name = 'debug_run_reactions_task_2025-10-09_15-02-57'
    # rs_run_name = 'beta20-reward-shaping-buffer6400'  # seh: beta20-reward-shaping-buffer6400 (typo of beta25)  re-beta25-reward-shaping
    # auxiliary_run_name = 'beta25-mutated-unlikelihood-buffer6400'

    task = 'vina-ALDH1'  # 'JNK3'
    sfn_run_name = 'debug_run_reactions_task_2025-11-24_19-38-05'
    # rtb_run_name = 'real-beta25-vanila'
    rs_run_name = 'hb3-beta25-sampling-rs-seed0'  # seh: beta20-reward-shaping-buffer6400 (typo of beta25)  re-beta25-reward-shaping
    auxiliary_run_name = 'hb3-beta25-re-sampling-relative-logp-e3-seed0'
    
    seen = {}
    
    for run_name in [rs_run_name, auxiliary_run_name]:  #, rs_run_name]:
        print(f"Running {run_name}...")
        if run_name == sfn_run_name:
            data = get_data_from_db(run_name, phase='final', unique=False)
        else:
            # data = get_data(run_name, task=task, phase='final', unique=False)
            data = get_data(run_name, task=task, phase='diverse_topk', unique=False)
        
        finder = AiZynthFinder(configfile="../../data/aizynthfinder/config.yaml")

        finder.stock.select("enamine")
        finder.expansion_policy.select("uspto")
        finder.filter_policy.select("uspto")

        success = []
        failed = []
        t_bar = tqdm(data[:100])
        for smiles in t_bar:
            # smiles = trj.smiles
            if smiles in seen.keys():
                rst = seen[smiles]
                if rst > 0:
                    success.append(smiles)
                else:
                    failed.append(smiles)
                t_bar.set_postfix(success=len(success), failed=len(failed))
                continue
            try:
                finder.target_smiles = smiles
                finder.tree_search()
                finder.build_routes()
                stat = finder.extract_statistics()  # 'number_of_steps'
                if stat["is_solved"] > 0:
                    success.append(smiles)
                    seen[smiles] = stat['number_of_steps']
                else:
                    failed.append(smiles)
                    seen[smiles] = -1
                t_bar.set_postfix(success=len(success), failed=len(failed))
            except Exception as e:
                seen[smiles] = -1
                failed.append(smiles)
                print(f"Failed to build routes for {smiles}: {e}")

        print(f"Success: {len(success)}")
        print(f"Average steps: {sum([seen[smiles] for smiles in success]) / len(success)}")
        print(f"Failed: {len(failed)}")