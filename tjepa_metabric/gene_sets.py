from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import requests


@dataclass(frozen=True)
class GeneSetLibrary:
    name: str
    gene_sets: dict[str, list[str]]


def _parse_enrichr_gene_set_lines(lines: Iterable[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Enrichr format varies by endpoint/library. We support:
        # - TERM<TAB>gene1,gene2,gene3...
        # - TERM<TAB><TAB>gene1<TAB>gene2<...>
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        term = parts[0]
        rest = [p for p in parts[1:] if p]
        genes: list[str] = []
        for chunk in rest:
            genes.extend([g for g in chunk.split(",") if g])
        genes = [g.strip() for g in genes if g.strip()]
        if genes:
            out[term] = genes
    return out


def download_enrichr_library(
    library_name: str = "Reactome_2022", *, timeout_s: int = 60
) -> GeneSetLibrary:
    """
    Download a gene set library from Enrichr.

    We default to Reactome because it's widely used and avoids MSigDB licensing issues.
    """
    url = "https://maayanlab.cloud/Enrichr/geneSetLibrary"
    resp = requests.get(
        url,
        params={"mode": "text", "libraryName": library_name},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    gene_sets = _parse_enrichr_gene_set_lines(resp.text.splitlines())
    if not gene_sets:
        raise ValueError(f"Downloaded library {library_name} but parsed 0 gene sets.")
    return GeneSetLibrary(name=library_name, gene_sets=gene_sets)
