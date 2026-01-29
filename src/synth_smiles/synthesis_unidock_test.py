from __future__ import annotations

import argparse
import os
from pathlib import Path

from rdkit import Chem

from gflownet.utils import sascore
from rxnflow.envs.action import Protocol, RxnActionType
from rxnflow.envs.reaction import BiReaction, Reaction, UniReaction
from rxnflow.envs.retrosynthesis import MultiRetroSyntheticAnalyzer, RetroSynthesisTree
from rxnflow.tasks.unidock_vina import VinaReward

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class SynthesizabilityEvaluator:
    """Lightweight synthesizability check reused from train_clean."""

    def __init__(
        self,
        num_workers: int = 4,
        invalid: float = 0.0,
        max_size: int = 50_000,
        use_retrosynthesis: bool = False,
        sa_threshold: float = 3.0,
        env: str = "stock",
        max_steps: int = 2,
    ):
        if use_retrosynthesis:
            env_dir = Path("../../data/envs/" + env)
            reaction_template_path = env_dir / "template.txt"
            building_block_path = env_dir / "building_block.smi"

            # Build RxnFlow retrosynthesis environment
            protocols: list[Protocol] = [
                Protocol("stop", RxnActionType.Stop),
                Protocol("firstblock", RxnActionType.FirstBlock),
            ]
            with reaction_template_path.open() as file:
                reaction_templates = [ln.strip() for ln in file.readlines()]
            for i, template in enumerate(reaction_templates):
                _rxn = Reaction(template)
                if _rxn.num_reactants == 1:
                    protocols.append(Protocol(f"unirxn{i}", RxnActionType.UniRxn, _rxn))
                elif _rxn.num_reactants == 2:
                    for block_is_first in [True, False]:  # order matters
                        rxn = BiReaction(template, block_is_first)
                        protocols.append(Protocol(f"birxn{i}_{block_is_first}", RxnActionType.BiRxn, rxn))

            with building_block_path.open() as file:
                blocks = [ln.split()[0] for ln in file.readlines()]

            self.retrosynthesis_analyzer = MultiRetroSyntheticAnalyzer.create(protocols, blocks, num_workers=num_workers)
        else:
            self.retrosynthesis_analyzer = None

        self.sa_threshold = sa_threshold
        self.invalid = invalid
        self._seen: dict[str, float | RetroSynthesisTree | None] = {}
        self._max = max_size
        self._max_steps = max_steps

    def get_synthesis(self, smiles: str) -> RetroSynthesisTree | None:
        if not self.retrosynthesis_analyzer:
            return None

        try:
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            return None

        if canonical_smiles in self._seen:
            return self._seen[canonical_smiles]  # type: ignore[return-value]

        self.retrosynthesis_analyzer.submit(0, smiles, self._max_steps, [])
        _, retro_tree = self.retrosynthesis_analyzer.result()[0]
        if len(self._seen) < self._max:
            self._seen[canonical_smiles] = retro_tree
        return retro_tree

    def score(self, smiles: str) -> float:
        try:
            mol = Chem.MolFromSmiles(smiles)
            canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        except Exception:
            return 0.0

        if canonical_smiles in self._seen:
            cached = self._seen[canonical_smiles]
            if self.retrosynthesis_analyzer:
                return 1.0 if cached else 0.0
            return float(cached < self.sa_threshold)  # type: ignore[arg-type]

        if self.retrosynthesis_analyzer:
            self.retrosynthesis_analyzer.submit(0, canonical_smiles, self._max_steps, [])
            _, retro_tree = self.retrosynthesis_analyzer.result()[0]
            score = 1.0 if retro_tree else 0.0
        else:
            try:
                sa = sascore.calculateScore(mol)
            except Exception:
                sa = 10.0
            score = float(sa < self.sa_threshold)
            retro_tree = sa  # reuse variable name for caching

        if len(self._seen) < self._max:
            self._seen[canonical_smiles] = retro_tree
        return score

    def score_batch(self, smiles_list: list[str]) -> list[float]:
        return [self.score(s) for s in smiles_list]


def main():

    smiles = "O=C(NCc1ccc2c(c1)CCOC2)C1CCc2ccccc2N1"
    smiles_list = ['CC1CCc2c(F)cccc2C1NC(=O)c1ccc2c(c1)C(=O)CC2'
                ,'CN1C(=O)Cc2c(C(=O)Nc3cccc4c3CCCC4=O)cccc21'
                ,'O=C1CCC(C(=O)NC2C3CCC2Cc2ccccc2C3)c2ccccc21']

    # Synthesizability
    synth_eval = SynthesizabilityEvaluator(
        use_retrosynthesis=True,
        env="stock_hb",
        max_steps=3,
    )
    synth_score = synth_eval.score_batch(smiles_list)

    # Vina / UniDock
    vina_receptor = "ALDH1"
    base_dir = Path("../../data/LIT-PCBA") / vina_receptor
    protein_pdb = base_dir / "protein.pdb"
    ref_ligand = base_dir / "ligand.mol2"

    if not protein_pdb.exists() or not ref_ligand.exists():
        raise FileNotFoundError(f"Missing docking inputs under {base_dir}. Expected protein.pdb and ligand.mol2.")

    vina = VinaReward(
        protein_pdb_path=protein_pdb,
        center=None,  # inferred from ref ligand
        ref_ligand_path=ref_ligand,
        search_mode="balance",
        num_workers=4,
    )
    vina_score = vina.run_smiles(smiles_list, save_path="./test_docking.sdf")[0]

    print("=== Single-sample environment check ===")
    print(f"SMILES           : {smiles}")
    print(f"Synthesizability : {synth_score}")
    print(f"Vina score       : {vina_score}")


if __name__ == "__main__":
    main()