from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class   BiomedicalStore:
    disease_to_proteins: dict[str, set[str]]
    disease_display: dict[str, str]
    protein_to_drugs: dict[str, set[str]]


class StoreBuilder:
    def __init__(self, disease_file: Path, protein_drug_file: Path):
        self.disease_file = disease_file
        self.protein_drug_file = protein_drug_file

    @staticmethod
    def _normalize(value: str) -> str:
        return value.strip()

    def _load_disease_to_protein(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = defaultdict(set)
        disease_display: dict[str, str] = {}
        with self.disease_file.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                disease = self._normalize(
                    row.get("disease")
                    or row.get("disease_name")
                    or row.get("Disease")
                    or ""
                )
                protein = self._normalize(
                    row.get("protein")
                    or row.get("protein_symbol")
                    or row.get("Protein")
                    or ""
                )
                if disease and protein:
                    disease_key = disease.lower()
                    mapping[disease_key].add(protein)
                    disease_display[disease_key] = disease
        self._disease_display = disease_display
        return mapping

    def _load_protein_to_drug(self) -> dict[str, set[str]]:
        mapping: dict[str, set[str]] = defaultdict(set)
        with self.protein_drug_file.open("r", encoding="utf-8") as fp:
            reader = csv.DictReader(fp)
            for row in reader:
                protein = self._normalize(
                    row.get("protein")
                    or row.get("protein_symbol")
                    or row.get("gene")
                    or row.get("Protein")
                    or ""
                )
                drug = self._normalize(
                    row.get("drug")
                    or row.get("drug_name")
                    or row.get("compound_name")
                    or row.get("Drug")
                    or ""
                )
                if protein and drug:
                    mapping[protein].add(drug)
        return mapping

    def build(self) -> BiomedicalStore:
        disease_to_proteins = self._load_disease_to_protein()
        return BiomedicalStore(
            disease_to_proteins=disease_to_proteins,
            disease_display=getattr(self, "_disease_display", {}),
            protein_to_drugs=self._load_protein_to_drug(),
        )
