"""SMILES representations used by LIG prediction models."""

from __future__ import annotations

from enum import Enum
import hashlib
import re
from typing import Iterable, Sequence


class MolecularRepresentation(str, Enum):
    """Supported molecular representations."""

    FINGERPRINT = "fingerprint"
    DESCRIPTORS = "descriptors"
    TOKENS = "tokens"


def _fallback_fingerprint(smiles: str, size: int = 2048) -> list[int]:
    """Create a deterministic, RDKit-independent Morgan-like fingerprint."""
    value = [0] * size
    smiles = smiles.strip()
    if not smiles:
        return value

    atoms = re.findall(r"\[[^\]]+\]|[A-Za-z]|[#%()\\/.=+\-]", smiles)
    features = atoms + [part for part in zip(atoms, atoms[1:])]
    for feature in features:
        if isinstance(feature, tuple):
            feature = "".join(feature)
        digest = hashlib.sha256(feature.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % size
        value[index] = 1
    return value


def _fallback_descriptors(smiles: str) -> list[float]:
    """A compact structural descriptor set that needs no external package."""
    clean = smiles.strip()
    return [
        float(len(clean)),
        float(len(re.findall(r"[A-Z]", clean))),
        float(clean.count("O")),
        float(clean.count("N")),
        float(clean.count("S")),
        float(clean.count("C")),
        float(clean.count("#")),
        float(clean.count("=")),
        float(clean.count("(")),
        float(clean.count(")")),
    ]


def _fallback_tokens(smiles: str, size: int = 768) -> list[float]:
    """Hash SMILES character tokens into a stable dense vector."""
    value = [0.0] * size
    clean = smiles.strip().lower()
    if not clean:
        return value
    tokens = re.findall(r"\[[^\]]+\]|[a-z]|[0-9]|[^a-z0-9]", clean)
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        value[int.from_bytes(digest[:4], "big") % size] += 1.0
    norm = sum(number * number for number in value) ** 0.5
    if norm:
        value = [number / norm for number in value]
    return value


def represent_smiles(
    smiles: str,
    representation: str | MolecularRepresentation = MolecularRepresentation.FINGERPRINT,
    **kwargs,
) -> list[int] | list[float]:
    """Convert one polymer SMILES string to a molecular representation."""
    if isinstance(representation, MolecularRepresentation):
        representation = representation.value
    if representation not in {item.value for item in MolecularRepresentation}:
        raise ValueError(
            "representation must be 'fingerprint', 'descriptors', or 'tokens'"
        )
    smiles = str(smiles).strip()
    if not smiles:
        raise ValueError("SMILES value cannot be empty")

    if representation == "fingerprint":
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem

            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            radius = int(kwargs.get("radius", 2))
            size = int(kwargs.get("size", 2048))
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule, radius, nBits=size
            )
            return list(fingerprint)
        except ImportError:
            return _fallback_fingerprint(smiles, int(kwargs.get("size", 2048)))

    if representation == "descriptors":
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors

            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            names = [
                "MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors",
                "NumHAcceptors", "NumRotatableBonds", "RingCount",
                "FractionCSP3", "NumAromaticRings",
            ]
            return [float(getattr(Descriptors, name)(molecule)) for name in names]
        except ImportError:
            return _fallback_descriptors(smiles)

    try:
        from transformers import AutoTokenizer

        model_name = kwargs.get("model_name", "DeepChem/ChemBERTa-77M-MTR")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        size = int(kwargs.get("size", 768))
        ids = tokenizer(smiles, return_tensors="np", padding=True, truncation=True)["input_ids"][0]
        vector = [0.0] * size
        for token_id in ids:
            vector[int(token_id) % size] += 1.0
        norm = sum(number * number for number in vector) ** 0.5
        return [number / norm for number in vector] if norm else vector
    except (ImportError, OSError):
        return _fallback_tokens(smiles, int(kwargs.get("size", 768)))


class SmilesEncoder:
    """Encode a sequence of polymer SMILES strings for model training."""

    def __init__(
        self,
        representation: str | MolecularRepresentation = MolecularRepresentation.FINGERPRINT,
        **representation_kwargs,
    ) -> None:
        self.representation = representation
        self.representation_kwargs = representation_kwargs

    def fit(self, smiles: Iterable[str]):
        # Encoding is stateless, so validation is the only fit-time operation.
        list(smiles)
        return self

    def transform(self, smiles: Iterable[str]) -> list[list[int] | list[float]]:
        return [
            represent_smiles(
                value,
                representation=self.representation,
                **self.representation_kwargs,
            )
            for value in smiles
        ]

    def fit_transform(self, smiles: Sequence[str]) -> list[list[int] | list[float]]:
        samples = list(smiles)
        return self.fit(samples).transform(samples)
