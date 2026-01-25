from rdkit import Chem
from rdkit.Chem import rdFMCS
from transformers import AutoTokenizer
import torch

from typing import Dict, List, Tuple
import re

from gflownet.utils import sascore
from genetic_operator import crossover as co
from genetic_operator import mutate as graph_ga_mutate
# from smiles_ga import mutate as smiles_ga_mutate


def mutate(smiles, synth_evaluator, mode="graph_ga", n_try: int=10):
    """
    Generate an unsynthesizable SMILES string by mutating the input SMILES string.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None

    for i in range(n_try):
        if mode == "graph_ga":
            mutated = graph_ga_mutate.mutate(mol, 1.0)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        if mutated is not None:
            mutated_smiles = Chem.MolToSmiles(mutated, isomericSmiles=False)
            if synth_evaluator.score(mutated_smiles):  # still synthesizable
                mol = mutated if (i+1) % 5 == 0 else Chem.MolFromSmiles(smiles)
            else:
                return None
    return None


def smiles_substring_diff(smiles1: str, smiles2: str):
    """
    Compare two SMILES in graph space (via RDKit).
    Returns the common part, and the unique parts of each molecule.
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES input")

    # Find Maximum Common Substructure (MCS)
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)

    # Substructure match indices
    match1 = set(mol1.GetSubstructMatch(mcs_mol))
    match2 = set(mol2.GetSubstructMatch(mcs_mol))

    # Mark atoms with mapping numbers (to preserve identity in SMILES)
    for a in mol1.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)
    for a in mol2.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + 1)

    # Canonical fragments: what's unique to each
    uniq1 = set(range(mol1.GetNumAtoms())) - match1
    uniq2 = set(range(mol2.GetNumAtoms())) - match2

    frag1 = Chem.MolFragmentToSmiles(mol1, uniq1, canonical=True) if uniq1 else None
    frag2 = Chem.MolFragmentToSmiles(mol2, uniq2, canonical=True) if uniq2 else None

    return {
        "smiles1": smiles1,
        "smiles2": smiles2,
        "common_substructure": Chem.MolToSmiles(mcs_mol),
        "only_in_1": frag1,
        "only_in_2": frag2
    }

def smiles_mask_with_tokenizer(smiles1: str, smiles2: str, tokenizer):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES")

    # MCS for common atoms
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match2 = set(mol2.GetSubstructMatch(mcs_mol))
    diff_atoms = {i: (0 if i in match2 else 1) for i in range(mol2.GetNumAtoms())}

    # Tokenize SMILES2
    tokens = tokenizer.tokenize(smiles2)
    input_ids = tokenizer(smiles2, add_special_tokens=True)["input_ids"]

    # Build mask at token level
    mask = []
    atom_counter = 0
    for tok_id, tok in zip(input_ids, tokenizer.convert_ids_to_tokens(input_ids)):
        if tok_id in [0,1,2]:  # special tokens
            mask.append(0)
        elif tok in ["C","N","O","F","Cl","Br","c","n","o","s","p"]:  # atom tokens
            mask.append(diff_atoms.get(atom_counter, 0))
            atom_counter += 1
        else:
            mask.append(0)  # ring digits, parentheses, '=', etc.
    return tokens, input_ids, mask


def diff_mask_molformer_bck(smiles1: str, smiles2: str, tokenizer):
    mol1, mol2 = Chem.MolFromSmiles(smiles1), Chem.MolFromSmiles(smiles2)
    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMILES")

    # --- 1. Find common substructure
    mcs = rdFMCS.FindMCS([mol1, mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match2 = set(mol2.GetSubstructMatch(mcs_mol))

    # --- 2. Build atom-level diff flags
    diff_atoms = {i: (0 if i in match2 else 1) for i in range(mol2.GetNumAtoms())}

    # --- 3. Add atom indices so we can map tokens
    for a in mol2.GetAtoms():
        a.SetAtomMapNum(a.GetIdx())
    mapped_smiles = Chem.MolToSmiles(mol2, canonical=True)

    # --- 4. Tokenize with MoLFormer
    enc = tokenizer(smiles2, add_special_tokens=True)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # --- 5. Build mask
    mask = []
    atom_counter = 0
    for tok_id, tok in zip(input_ids, tokens):
        if tok_id in [0,1,2]:   # special tokens
            mask.append(0)
        elif tok in ["C","N","O","F","Cl","Br","c","n","o","s","p"]:  # atom tokens
            mask.append(diff_atoms.get(atom_counter, 0))
            atom_counter += 1
        else:
            mask.append(0)  # parentheses, digits, bonds
    return tokens, input_ids, mask


from typing import Dict, List, Tuple, Set
import re
from rdkit import Chem
from rdkit.Chem import rdFMCS

# ---------------------------
# MCS (robust, ring-safe)
# ---------------------------
def _graph_diff_mcs(mol1: Chem.Mol, mol2: Chem.Mol) -> Tuple[Set[int], Set[Tuple[int,int]]]:
    """
    Returns:
      changed_atoms2: atom idx set in mol2 not included in the MCS
      changed_bonds2: bonds (i,j) in mol2 that are new or have different order vs mol1
    MCS settings:
      - AtomCompare = CompareElements (require same element)
      - BondCompare = CompareAny (avoid aromatic vs Kekulé false diffs inside MCS)
      - RingMatchesRingOnly / CompleteRingsOnly conservative to avoid partial ring artifacts
    """
    params = rdFMCS.MCSParameters()
    params.AtomCompare = rdFMCS.AtomCompare.CompareElements
    params.BondCompare  = rdFMCS.BondCompare.CompareAny
    params.RingMatchesRingOnly = True
    params.CompleteRingsOnly   = True
    # params.MatchValences = True   # optional, enable if you want stricter mapping

    mcs = rdFMCS.FindMCS([mol1, mol2], parameters=params)
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString) if mcs.smartsString else None

    if not mcs_mol:
        # no overlap: everything in mol2 is changed
        changed_atoms2 = set(range(mol2.GetNumAtoms()))
        changed_bonds2 = {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
                          for b in mol2.GetBonds()}
        return changed_atoms2, changed_bonds2

    match2 = set(mol2.GetSubstructMatch(mcs_mol))
    changed_atoms2 = set(range(mol2.GetNumAtoms())) - match2

    def bonds_as_pairs(m):
        d = {}
        for b in m.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            d[tuple(sorted((i, j)))] = b.GetBondType()
        return d

    b1, b2 = bonds_as_pairs(mol1), bonds_as_pairs(mol2)
    changed_bonds2 = set()
    for (i, j), t2 in b2.items():
        t1 = b1.get((i, j))
        if t1 is None or t1 != t2:
            changed_bonds2.add((i, j))

    return changed_atoms2, changed_bonds2


# ---------------------------------------------
# Fallback: re-include obvious exocyclic matches
# ---------------------------------------------
def _fallback_reinclude_obvious(mol1: Chem.Mol, mol2: Chem.Mol, changed_atoms2: Set[int]) -> Set[int]:
    """
    Fix common MCS false positives: if an atom in mol2 has a clear "twin" in mol1
    (same element, same heavy-atom degree, same aromatic and ring membership), we
    treat it as common and remove from 'changed'.
    """
    # Build signature multimap for mol1
    def sigs(m):
        S = {}
        for a in m.GetAtoms():
            if a.GetSymbol() == 'H':
                continue
            sym  = a.GetSymbol()
            deg  = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() != 'H')
            aro  = int(a.GetIsAromatic())
            ring = int(a.IsInRing())
            S.setdefault((sym, deg, aro, ring), 0)
        return S

    sig1 = {}
    for a in mol1.GetAtoms():
        if a.GetSymbol() == 'H': continue
        sym  = a.GetSymbol()
        deg  = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() != 'H')
        aro  = int(a.GetIsAromatic())
        ring = int(a.IsInRing())
        sig1.setdefault((sym, deg, aro, ring), 0)
        sig1[(sym, deg, aro, ring)] += 1

    # Try to re-include mol2 atoms whose signature definitely exists in mol1
    rescued = set()
    for idx in changed_atoms2:
        a = mol2.GetAtomWithIdx(idx)
        if a.GetSymbol() == 'H':
            continue
        sym  = a.GetSymbol()
        deg  = sum(1 for nb in a.GetNeighbors() if nb.GetSymbol() != 'H')
        aro  = int(a.GetIsAromatic())
        ring = int(a.IsInRing())
        if sig1.get((sym, deg, aro, ring), 0) > 0:
            # heuristic: require at least one neighbor in mol2 that is unchanged (anchors it)
            if any((nb.GetIdx() not in changed_atoms2) for nb in a.GetNeighbors()):
                rescued.add(idx)

    return changed_atoms2 - rescued


# ---------------------------------------------------
# Annotate ORIGINAL smiles2 with per-atom text indices
# ---------------------------------------------------
_ATOM_TWO = {"Cl","Br","Si","Se","As"}   # extend if needed
_AROM_TWO = {"se","as"}                   # rare but supported
_AROM_ONE = set(list("bcno ps".replace(" ", "")))  # {'b','c','n','o','p','s'}
_ELEM_ONE = set(list("BCNOSPFIHKV"))      # extend if you want more (Al, Mg, etc.)

def _annotate_smiles_with_text_indices(smiles: str):
    """
    Convert each atom in the ORIGINAL text into [X:idx] (idx = text-order),
    and record (idx, start, end) char span in ORIGINAL text.
    """
    out, spans = [], []
    i, k, L = 0, 0, len(smiles)

    while i < L:
        ch = smiles[i]

        if ch == '[':
            j = i + 1
            while j < L and smiles[j] != ']':
                j += 1
            if j >= L:
                out.append(smiles[i:])
                break
            inside = re.sub(r':\d+', '', smiles[i+1:j])  # strip any existing :map
            out.append('[' + inside + f':{k}]')
            spans.append((k, i, j+1))
            i, k = j + 1, k + 1
            continue

        if i + 1 < L:
            two = smiles[i:i+2]
            if two in _ATOM_TWO or two in _AROM_TWO:
                out.append(f'[{two}:{k}]')
                spans.append((k, i, i+2))
                i, k = i + 2, k + 1
                continue

        if ch in _AROM_ONE:
            out.append(f'[{ch}:{k}]')
            spans.append((k, i, i+1))
            i, k = i + 1, k + 1
            continue

        if ch in _ELEM_ONE:
            out.append(f'[{ch}:{k}]')
            spans.append((k, i, i+1))
            i, k = i + 1, k + 1
            continue

        out.append(ch)
        i += 1

    return ''.join(out), spans  # spans in ORIGINAL text coordinates


# -----------------------------------------
# Token offsets (requires a fast HF tokenizer)
# -----------------------------------------
def _build_token_offsets(tokenizer, text: str):
    enc = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    ids  = enc["input_ids"]
    offs = enc["offset_mapping"]   # (start,end) per token in ORIGINAL text; specials → (0,0)
    toks = tokenizer.convert_ids_to_tokens(ids)
    return ids, toks, offs


# =========================================
# Main: graph diff → MoLFormer token mask
# =========================================
def diff_mask_molformer(smiles1: str, smiles2: str, tokenizer, mark_bonds: bool=False, return_ones_after_deviation: bool=False):
    """
    Returns:
      {
        "input_ids": [...],
        "tokens": [...],
        "mask":  [...],  # 1 if token overlaps changed atom spans (and bonds if mark_bonds=True), else 0
        "changed_atoms2_rdidx": [...],
        "text_atom_indices_changed": [...],
        "annotated_smiles2": str
      }

    Key features:
      * Differences computed in graph space (MCS) → avoids c vs C=C false positives.
      * Consistent indices: we MCS against an ANNOTATED mol2 built from ORIGINAL text,
        then map RDKit indices → text-order indices via AtomMapNum.
      * Fallback heuristic to re-include obvious exocyclic matches (fixes 'N' marked spuriously).
      * Token mask uses HF tokenizer offset mappings (no guessing).
    """
    mol1 = Chem.MolFromSmiles(smiles1)
    if mol1 is None:
        raise ValueError("Invalid SMILES1")

    # 1) Annotate ORIGINAL smiles2 to tie atoms to text positions
    annotated_text, spans = _annotate_smiles_with_text_indices(smiles2)
    mol2_annot = Chem.MolFromSmiles(annotated_text)
    if mol2_annot is None:
        raise ValueError("Invalid SMILES2 (failed after annotation)")

    # 2) Graph-level diffs against the annotated mol2
    changed_atoms2_rdidx, changed_bonds2 = _graph_diff_mcs(mol1, mol2_annot)

    # 3) Fallback clean-up (reduces spurious flags like the exocyclic N)
    changed_atoms2_rdidx = _fallback_reinclude_obvious(mol1, mol2_annot, changed_atoms2_rdidx)

    # 4) Map RDKit idx → text idx using AtomMapNum on mol2_annot
    text_changed = set()
    for a in mol2_annot.GetAtoms():
        t_idx = a.GetAtomMapNum()
        if t_idx >= 0 and a.GetIdx() in changed_atoms2_rdidx:
            text_changed.add(t_idx)

    # 5) Tokenize ORIGINAL smiles2 and build mask by offset overlap
    ids, toks, offs = _build_token_offsets(tokenizer, smiles2)

    # Char spans of changed atoms in ORIGINAL text
    changed_char_spans = [(s, e) for (k, s, e) in spans if k in text_changed]

    def _overlaps(a0, a1, b0, b1):  # [a0,a1) with [b0,b1)
        return not (a1 <= b0 or b1 <= a0)

    mask = []
    for (tok_start, tok_end), tid, tok in zip(offs, ids, toks):
        if tid in (0, 1, 2):  # special tokens
            mask.append(0)
            continue

        flag = 0
        for (s, e) in changed_char_spans:
            if _overlaps(tok_start, tok_end, s, e):
                flag = 1
                break

        if mark_bonds and flag == 0 and tok in ('=', '#'):  #('=', '#'):
            # Conservative: mark bond tokens if you want, or implement char↔(i,j) mapping for precision
            flag = 1

        mask.append(flag)

    return {
        "input_ids": ids,
        "tokens": toks,
        "mask": mask,
        "changed_atoms2_rdidx": sorted(changed_atoms2_rdidx),
        "text_atom_indices_changed": sorted(text_changed),
        "annotated_smiles2": annotated_text,
    }


if __name__ == "__main__":
    # Example: Ethanol vs Acetaldehyde
    # print(smiles_substring_diff("CCO", "CC=O"))
    s1 = "NCC1(c2cccc(C(F)(F)F)c2)CCC1"
    s2 = "NC(O)C1(c2cccc(C(F)(F)F)c2)CCC1"

    mol = Chem.MolFromSmiles(s1)

    # for i in range(100):
    #     mutated = mu.mutate(mol, 1.0)
    #     if mutated is not None:
    #         if sascore.calculateScore(mutated) < 3.5:  # still synthesizable
    #             mol = mutated if (i+1) % 5 == 0 else Chem.MolFromSmiles(s1)
    #         else:
    #             print(sascore.calculateScore(mutated))
    #             break

    print(s1, Chem.MolToSmiles(mol, isomericSmiles=False))
    s2 = 'NCC1(C2C=C(C(F)(F)F)C=CC2)CCC1'  #'NCC1(C2=CC=CC2C(F)(F)F)CCC1'  # Chem.MolToSmiles(mutated, isomericSmiles=False)  # NCC1(C2C=C(C(F)(F)F)C=CC2)CCC1
    print(s2)

    tokenizer = AutoTokenizer.from_pretrained(
        "ibm-research/MoLFormer-XL-both-10pct",
        trust_remote_code=True
    )

    print(s1, s2)
    tokens, ids, mask = diff_mask_molformer_bck(s1, s2, tokenizer)
    res = diff_mask_molformer(s1, s2, tokenizer)
    res2 = diff_mask_molformer(s1, s2, tokenizer, mark_bonds=True)
    
    print(tokens)
    print(ids)
    print(mask)
    print(res)