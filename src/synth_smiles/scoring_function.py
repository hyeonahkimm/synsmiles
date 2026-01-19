import os
import glob
import numpy as np
from tdc import Oracle, Evaluator

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit import RDLogger
from rdkit.Chem import QED
from rdkit.Chem import Mol as RDMol
RDLogger.DisableLog('rdApp.*')

import subprocess
import multiprocessing

import torch
import torch_geometric.data as gd
from gflownet.models import bengio2021flow
from gflownet.utils.misc import get_worker_device
from gflownet.tasks.seh_frag_moo import aux_tasks
from gflownet.utils.conditioning import MultiObjectiveWeightedPreferences


def int_div(smiles):
    evaluator = Evaluator(name = 'Diversity')
    return evaluator(smiles)


def get_scores(smiles, mode="qed", n_process=1, models=None, pref_cond=None, vina=None, hist= {}):
    smiles_groups = []
    group_size = len(smiles) / n_process
    for i in range(n_process):
        smiles_groups += [smiles[int(i * group_size):int((i + 1) * group_size)]]

    temp_data = []
    if models is None:
        pool = multiprocessing.Pool(processes = n_process)
        for index in range(n_process):
            temp_data.append(pool.apply_async(get_scores_subproc, args=(smiles_groups[index], mode, vina)))
        pool.close()
        pool.join()
        scores = []
        for index in range(n_process):
            scores += temp_data[index].get()
    else:
        return get_scores_subproc(smiles, mode, models, pref_cond=pref_cond, vina=vina, hist=hist)

    return scores


def safe(f, x, default):
    try:
        return f(x)
    except Exception:
        return default


def mol2sascore(mols: list[RDMol], default=10):
    sas = torch.tensor([safe(sascore.calculateScore, mol, default) for mol in mols])
    sas = (10 - sas) / 9  # Turn into a [0-1] reward
    return sas


def calc_seh_reward(graphs: list[gd.Data]):
    seh_model = bengio2021flow.load_original_model()
    device = get_worker_device()
    seh_model.to(device)
    seh_model.eval()

    batch = gd.Batch.from_data_list([i for i in graphs if i is not None]).to(device)
    preds = seh_model(batch).reshape((-1,)).data.cpu() / 8
    preds[preds.isnan()] = 0
    return preds.clip(1e-4, 100).reshape((-1,))


def mol2seh(mols: list[RDMol], default=0):
    """Calculate SEH scores for a list of molecules using the pretrained model."""
    # Load SEH model if not provided
    model = bengio2021flow.load_original_model()
    device = get_worker_device()
    model.to(device)
    model.eval()
    
    scores = []
    for mol in mols:
        if mol is None:
            scores.append(default)
            continue
            
        # Convert molecule to graph
        try:
            graph = bengio2021flow.mol2graph(mol)
            # Compute SEH score
            device = model.device if hasattr(model, "device") else get_worker_device()
            batch = gd.Batch.from_data_list([graph]).to(device)
            
            with torch.no_grad():
                pred = model(batch).reshape((-1,)).data.cpu() / 8
                pred[pred.isnan()] = 0
                pred = pred.clip(1e-4, 100).item()
        
        except Exception as e:
            scores.append(default)
            continue
        
        scores.append(pred)
    
    return scores


def get_scores_subproc(smiles, mode, models=None, default=0.0, pref_cond=None, vina=None, hist= {}):
    scores = []
    mols = [MolFromSmiles(s) for s in smiles]
    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    if mode == "qed":
        for i in range(len(smiles)):
            if mols[i]:
                scores.append([QED.qed(mols[i])])
            else:
                scores.append([default])

    elif mode == "seh":
        scores = mol2seh(mols, default=default)

    elif mode == "jnk3":
        oracle = Oracle(name='JNK3')
        for i in range(len(smiles)):
            if mols[i] != None:
                try:
                    scores += safe(oracle, [smiles[i]], 0.0)
                except Exception as e:
                    scores += [0.0]
            else:
                scores += [0.0]
    
    elif mode == "vina":
        assert vina is not None
        
        vina_scores = []
        qed_scores = []
        unseen_idx = []
        for i in range(len(smiles)):
            if mols[i]:
                canonical_s = MolToSmiles(mols[i], isomericSmiles=False)
                if MolToSmiles(mols[i]) in hist.keys():
                    vina_scores.append(hist[canonical_s]['vina'])
                    qed_scores.append(hist[canonical_s]['qed'])
                else:
                    unseen_idx.append(i)
                    vina_scores.append(0)
                    qed_scores.append(QED.qed(mols[i]))
            else:
                vina_scores.append(0)
                qed_scores.append(default)
        unseen_vina_scores = vina.run_smiles([smiles[i] for i in unseen_idx], save_path=None)

        for i, v in enumerate(unseen_vina_scores):
            vina_scores[unseen_idx[i]] = v
        vina_scores = [v if v >= -20 else 0 for v in vina_scores]
        r = [0.5 * (-0.1 * min(vina_score, 0)) + 0.5 * qed_score for vina_score, qed_score in zip(vina_scores, qed_scores)]
        scores = list(zip(r, vina_scores, qed_scores))  # (r, vina_score, qed_score)

    else:
        raise Exception("Scoring function undefined!")


    return scores

