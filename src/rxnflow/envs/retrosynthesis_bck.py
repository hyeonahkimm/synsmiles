import concurrent.futures
import copy
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from typing_extensions import Self

from pathlib import Path
from rdkit import Chem
import numpy as np
from numpy.typing import NDArray

from rxnflow.envs.action import Protocol, RxnAction, RxnActionType
from rxnflow.envs.reaction import BiReaction, Reaction, UniReaction


class RetroSynthesisTree:
    smi: str
    branches: list[tuple[RxnAction, Self]]

    def __init__(self, smi: str, branches: list[tuple[RxnAction, Self]] | None = None):
        self.smi = smi
        self.branches = branches if branches is not None else []
        self._height: int | None = None

    @property
    def is_leaf(self) -> bool:
        return len(self.branches) == 0

    def __len__(self):
        return len(self.branches)

    def height(self) -> int:
        if self._height is None:
            self._height = max(self.iteration_depth(0))
        return self._height

    def iteration_depth(self, prev_len: int = 0) -> Iterable[int]:
        if self.is_leaf:
            yield prev_len
        elif not self.is_leaf:
            for _, subtree in self.branches:
                yield from subtree.iteration_depth(prev_len + 1)

    def print(self, indent=0):
        print(" " * indent + "SMILES: " + self.smi)
        for action, child in self.branches:
            print(" " * (indent + 2) + "- ACTION:", action)
            if not child.is_leaf:
                child.print(indent + 4)

    def iteration(self, prev_traj: list[RxnAction] | None = None) -> Iterable[list[RxnAction]]:
        prev_traj = prev_traj if prev_traj else []
        if self.is_leaf:
            yield prev_traj
        else:
            for action, subtree in self.branches:
                yield from subtree.iteration(prev_traj + [action])


class RetroSyntheticAnalyzer:
    def __init__(
        self,
        protocols: list[Protocol],
        blocks: list[str],
        approx: bool = True,
        max_decomposes: int = 2,
    ):
        self.protocols: list[Protocol] = protocols
        self.approx: bool = approx  # Fast analyzing
        self.__cache_success: Cache = Cache(100_000)
        self.__cache_fail: Cache = Cache(1_000_000)
        self.max_decomposes: int = max_decomposes

        # For Fast Search
        self.__block_search: dict[int, dict[str, int]] = {}
        for idx, smi in enumerate(blocks):
            smi_len = len(smi)
            self.__block_search.setdefault(smi_len, dict())[smi] = idx

        # temporary
        self.__min_depth: int
        self.__max_depth: int

    def run(
        self,
        mol: str | Chem.Mol,
        max_rxns: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:

        if isinstance(mol, Chem.Mol):
            smiles = Chem.MolToSmiles(mol)
        else:
            smiles = mol

        self.__max_depth = self.__min_depth = max_rxns + 1  # 1: AddFirstBlock
        if known_branches is not None:
            for _, tree in known_branches:
                self.__min_depth = min(self.__min_depth, min(tree.iteration_depth()) + 1)
        res = self.__dfs(smiles, 1, known_branches)
        del self.__max_depth, self.__min_depth
        return res

    def block_search(self, smi: str) -> int | None:
        assert isinstance(smi, str)
        prefix_block_set = self.__block_search.get(len(smi), None)
        if prefix_block_set is None:
            return None
        return prefix_block_set.get(smi, None)

    # For entire run
    def from_cache(self, smi: str, depth: int) -> tuple[bool, RetroSynthesisTree | None]:
        is_cached, _ = self.__cache_fail.get(smi, depth)
        if is_cached:
            return True, None
        is_cached, cached_tree = self.__cache_success.get(smi, depth)
        if is_cached:
            return True, cached_tree
        return False, None

    def to_cache(self, smi: str, depth: int, cache: RetroSynthesisTree | None):
        if cache is None:
            self.__cache_fail.update(smi, depth, None)
        else:
            self.__cache_success.update(smi, depth, cache)

    # Check tree depth
    def check_depth(self, depth: int) -> bool:
        if depth > self.__max_depth:
            return False
        if self.approx and (depth > self.__min_depth):
            return False
        return True

    def __dfs(
        self,
        smiles: str,
        depth: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]] | None = None,
    ) -> RetroSynthesisTree | None:
        # Check state
        if (not self.check_depth(depth)) or (len(smiles) == 0):
            return None

        # Load cache
        is_cached, cached_tree = self.from_cache(smiles, depth)
        if is_cached:
            return cached_tree

        # convert mol
        mol: Chem.Mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        if known_branches is None:
            known_branches = []
        known_protocols: set[str] = set(action.protocol for action, _ in known_branches)
        branches: list[tuple[RxnAction, RetroSynthesisTree]] = known_branches.copy()

        # Run
        is_block = False
        for protocol in self.protocols:
            # pass if the protocol is in known branches
            if protocol.name in known_protocols:
                continue

            # run retrosynthesis
            if protocol.action is RxnActionType.FirstBlock:
                block_idx = self.block_search(smiles)
                if block_idx is not None:
                    bck_action = RxnAction(RxnActionType.FirstBlock, protocol.name, smiles, block_idx)
                    branches.append((bck_action, RetroSynthesisTree("")))
                    self.__min_depth = depth
                    is_block = True
            elif protocol.action is RxnActionType.UniRxn:
                if not self.check_depth(depth + 1):
                    continue
                for child_smi, *_ in protocol.rxn.reverse_smi(mol)[: self.max_decomposes]:
                    child_tree = self.__dfs(child_smi, depth + 1)
                    if child_tree is not None:
                        bck_action = RxnAction(RxnActionType.UniRxn, protocol.name)
                        branches.append((bck_action, child_tree))
            elif protocol.action is RxnActionType.BiRxn:
                if not self.check_depth(depth + 1):
                    continue
                for child_smi, block_smi in protocol.rxn.reverse_smi(mol)[: self.max_decomposes]:
                    block_idx = self.block_search(block_smi)
                    if block_idx is not None:
                        child_tree = self.__dfs(child_smi, depth + 1)
                        if child_tree is not None:
                            bck_action = RxnAction(RxnActionType.BiRxn, protocol.name, block_smi, block_idx)
                            branches.append((bck_action, child_tree))

        # return None if retrosynthetically inaccessible
        if len(branches) == 0:
            result = None
        else:
            result = RetroSynthesisTree(smiles, branches)

        # update cache
        # if self.approx is True, we don't save cache for building blocks
        if not (self.approx and is_block):
            self.to_cache(smiles, depth, result)
        return result


class Cache:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.cache_valid: dict[str, tuple[int, RetroSynthesisTree]] = {}
        self.cache_invalid: dict[str, int] = {}

    def update(self, smiles: str, height: int, tree: RetroSynthesisTree | None):
        if tree is not None:
            flag, cache = self.get(smiles, height)
            if flag is False:
                if len(self.cache_valid) >= self.max_size:
                    self.cache_valid.popitem()
                self.cache_valid[smiles] = (height, tree)
        else:
            self.cache_invalid[smiles] = max(self.cache_invalid.get(smiles, -1), height)

    def get(self, smiles: str, height: int) -> tuple[bool, RetroSynthesisTree | None]:
        cache = self.cache_valid.get(smiles, None)
        if cache is not None:
            cached_height, cached_tree = cache
            if height <= cached_height:
                return True, cached_tree
        cached_height = self.cache_invalid.get(smiles, -1)
        if height <= cached_height:
            return True, None
        else:
            return False, None


class MultiRetroSyntheticAnalyzer:
    def __init__(self, analyzer, num_workers: int = 4):
        self.pool = ProcessPoolExecutor(num_workers, initializer=MultiRetroSyntheticAnalyzer._init_worker, initargs=(analyzer,))
        self.futures = []

    @classmethod
    def create(
        cls,
        protocols: list[Protocol],
        blocks: list[str],
        approx: bool = True,
        max_decomposes: int = 2,
        num_workers: int = 4,
    ):
        analyzer = RetroSyntheticAnalyzer(protocols, blocks, approx, max_decomposes)
        return cls(analyzer, num_workers)

    @staticmethod
    def _init_worker(base_analyzer):
        global analyzer
        analyzer = copy.deepcopy(base_analyzer)

    def init(self):
        self.result()

    def terminate(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def submit(
        self,
        key: int,
        mol: str | Chem.Mol,
        max_rxns: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]],
    ):
        self.futures.append(self.pool.submit(self._worker, key, mol, max_rxns, known_branches))

    def result(self) -> list[tuple[int, RetroSynthesisTree]]:
        try:
            done, _ = concurrent.futures.wait(self.futures, return_when=concurrent.futures.FIRST_EXCEPTION)
            result = [future.result() for future in done]
            self.futures = []
            return result
        except Exception as e:
            print("Error during Retrosynthesis analysis")
            raise e

    @staticmethod
    def _worker(
        key: int,
        mol: str | Chem.Mol,
        max_step: int,
        known_branches: list[tuple[RxnAction, RetroSynthesisTree]],
    ) -> tuple[int, RetroSynthesisTree]:
        global analyzer
        res = analyzer.run(mol, max_step, known_branches)
        return key, res


if __name__ == "__main__":

    env_dir = env_dir = Path('../../data/envs/stock')
    reaction_template_path = env_dir / "template.txt"
    building_block_path = env_dir / "building_block.smi"
    pre_computed_building_block_mask_path = env_dir / "bb_mask.npy"
    pre_computed_building_block_fp_path = env_dir / "bb_fp_2_1024.npy"
    pre_computed_building_block_desc_path = env_dir / "bb_desc.npy"

    # set protocol
    protocols: list[Protocol] = []
    protocols.append(Protocol("stop", RxnActionType.Stop))
    protocols.append(Protocol("firstblock", RxnActionType.FirstBlock))
    with reaction_template_path.open() as file:
        reaction_templates = [ln.strip() for ln in file.readlines()]
    for i, template in enumerate(reaction_templates):
        _rxn = Reaction(template)
        if _rxn.num_reactants == 1:
            rxn = UniReaction(template)
            protocols.append(Protocol(f"unirxn{i}", RxnActionType.UniRxn, _rxn))
        elif _rxn.num_reactants == 2:
            for block_is_first in [True, False]:  # this order is important
                rxn = BiReaction(template, block_is_first)
                protocols.append(Protocol(f"birxn{i}_{block_is_first}", RxnActionType.BiRxn, rxn))
    protocol_dict: dict[str, Protocol] = {protocol.name: protocol for protocol in protocols}
    stop_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.Stop]
    firstblock_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.FirstBlock]
    unirxn_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.UniRxn]
    birxn_list: list[Protocol] = [p for p in protocols if p.action is RxnActionType.BiRxn]

    # set building blocks
    with building_block_path.open() as file:
        lines = file.readlines()
        building_blocks = [ln.split()[0] for ln in lines]
        building_block_ids = [ln.strip().split()[1] for ln in lines]
    blocks: list[str] = building_blocks

    # set precomputed building block feature
    block_fp = np.load(pre_computed_building_block_fp_path)
    block_prop = np.load(pre_computed_building_block_desc_path)

    # set block mask
    block_mask: NDArray[np.bool_] = np.load(pre_computed_building_block_mask_path)
    birxn_block_indices: dict[str, np.ndarray] = {}
    for i, protocol in enumerate(birxn_list):
        birxn_block_indices[protocol.name] = np.where(block_mask[i])[0]
    num_total_actions = (
        1 + len(unirxn_list) + sum(indices.shape[0] for indices in birxn_block_indices.values())
    )

    # import pdb; pdb.set_trace()  # Commented out for multiprocessing
    retro_analyzer = MultiRetroSyntheticAnalyzer.create(protocols, blocks, num_workers=4)

    retro_analyzer.submit(0, "NCC1(c2cccc(C(F)(F)F)c2)CCC1", 2, [])  # NCC1(c2cccc(C(F)(F)F)c2)CCC1
    _, retro_tree = retro_analyzer.result()[0]  # retro_tree is None if retrosynthetically inaccessible
    print(retro_tree)