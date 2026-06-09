"""
Concept catalog + per-source query map for the Scientific Trend Forecaster.

This is the single editable source of truth for *what* the model tracks. Each entry is
one research topic with the query string each data source needs. To add or remove a
topic, edit the CONCEPTS list below -- nothing else in the notebook hard-codes topics.

Fields per concept
------------------
key            : short unique slug (used in filenames / dataframes)
name           : human-readable display name
domain         : broad field, used for grouping + a categorical model feature
openalex_search: query for OpenAlex `title_and_abstract.search` (works for emerging
                 topics that have no formal concept ID yet; noisier than concept tags,
                 noted as a caveat in the notebook). Use AND/OR/quotes as needed.
arxiv_query    : arXiv API `search_query` string, or None if arXiv barely covers the
                 field (e.g. clinical medicine). arXiv leads journals ~6-18 months.
biorxiv_terms  : list of lowercase substrings matched against bioRxiv preprint titles,
                 or None for non-bio topics. bioRxiv data starts 2013.
nih_term       : NIH RePORTER text search term (biomedical funding), or None.
nsf_query      : NSF Awards API keyword (physical sciences / CS / engineering), or None.
wikipedia_title: exact Wikipedia article title for the pageviews API, or None.
                 Pageviews REST API only returns data from 2015-07 onward.
s2_query       : Semantic Scholar query used to fetch a SPECTER topic embedding.
                 Defaults to `name` when None.

Coverage is intentionally cross-domain so the model can compare "what grows next"
across medicine, AI/CS, materials, climate/energy, quantum, physics and biology.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class Concept:
    key: str
    name: str
    domain: str
    openalex_search: str
    arxiv_query: Optional[str] = None
    biorxiv_terms: Optional[List[str]] = None
    nih_term: Optional[str] = None
    nsf_query: Optional[str] = None
    wikipedia_title: Optional[str] = None
    s2_query: Optional[str] = None

    @property
    def s2(self) -> str:
        return self.s2_query or self.name


# --------------------------------------------------------------------------------------
# The catalog (~36 topics across 8 domains)
# --------------------------------------------------------------------------------------
CONCEPTS: List[Concept] = [
    # ---------------------------- Neuroscience ----------------------------
    Concept("alzheimers", "Alzheimer's disease", "Neuroscience",
            openalex_search='"alzheimer"',
            arxiv_query=None, biorxiv_terms=["alzheimer"],
            nih_term="Alzheimer disease", nsf_query=None,
            wikipedia_title="Alzheimer's_disease"),
    Concept("parkinsons", "Parkinson's disease", "Neuroscience",
            openalex_search='"parkinson"',
            biorxiv_terms=["parkinson"], nih_term="Parkinson disease",
            wikipedia_title="Parkinson's_disease"),
    Concept("optogenetics", "Optogenetics", "Neuroscience",
            openalex_search='"optogenetic"',
            biorxiv_terms=["optogenetic"], nih_term="optogenetics",
            wikipedia_title="Optogenetics"),
    Concept("brain_organoids", "Brain organoids", "Neuroscience",
            openalex_search='"brain organoid" OR "cerebral organoid"',
            biorxiv_terms=["brain organoid", "cerebral organoid"],
            nih_term="brain organoid", wikipedia_title="Cerebral_organoid"),
    Concept("connectomics", "Connectomics", "Neuroscience",
            openalex_search='"connectome" OR "connectomics"',
            biorxiv_terms=["connectome"], nih_term="connectome",
            nsf_query="connectomics", wikipedia_title="Connectomics"),

    # ---------------------------- Oncology ----------------------------
    Concept("cancer_immunotherapy", "Cancer immunotherapy", "Oncology",
            openalex_search='"cancer immunotherapy" OR "tumor immunotherapy"',
            biorxiv_terms=["immunotherapy"], nih_term="cancer immunotherapy",
            wikipedia_title="Cancer_immunotherapy"),
    Concept("car_t", "CAR-T cell therapy", "Oncology",
            openalex_search='"CAR-T" OR "chimeric antigen receptor"',
            biorxiv_terms=["chimeric antigen receptor", "car-t", "car t"],
            nih_term="chimeric antigen receptor", wikipedia_title="Chimeric_antigen_receptor_T_cell"),
    Concept("liquid_biopsy", "Liquid biopsy", "Oncology",
            openalex_search='"liquid biopsy" OR "circulating tumor DNA"',
            biorxiv_terms=["liquid biopsy", "circulating tumor dna"],
            nih_term="liquid biopsy", wikipedia_title="Liquid_biopsy"),
    Concept("immune_checkpoint", "Immune checkpoint inhibitors", "Oncology",
            openalex_search='"immune checkpoint" OR "PD-1" OR "PD-L1"',
            biorxiv_terms=["immune checkpoint", "pd-1", "pd-l1"],
            nih_term="immune checkpoint inhibitor", wikipedia_title="Checkpoint_inhibitor"),

    # ---------------------------- Genomics / Biotech ----------------------------
    Concept("crispr", "CRISPR gene editing", "Genomics",
            openalex_search='"CRISPR"',
            biorxiv_terms=["crispr"], nih_term="CRISPR",
            nsf_query="CRISPR", wikipedia_title="CRISPR"),
    Concept("single_cell_rna", "Single-cell RNA sequencing", "Genomics",
            openalex_search='"single-cell RNA" OR "scRNA-seq" OR "single cell sequencing"',
            biorxiv_terms=["single-cell rna", "scrna-seq", "single cell rna"],
            nih_term="single cell RNA sequencing", wikipedia_title="Single-cell_sequencing"),
    Concept("spatial_transcriptomics", "Spatial transcriptomics", "Genomics",
            openalex_search='"spatial transcriptomic"',
            biorxiv_terms=["spatial transcriptomic"], nih_term="spatial transcriptomics",
            wikipedia_title="Spatial_transcriptomics"),
    Concept("mrna_vaccines", "mRNA vaccines", "Genomics",
            openalex_search='"mRNA vaccine"',
            biorxiv_terms=["mrna vaccine"], nih_term="mRNA vaccine",
            wikipedia_title="MRNA_vaccine"),
    Concept("microbiome", "Microbiome", "Genomics",
            openalex_search='"microbiome"',
            biorxiv_terms=["microbiome"], nih_term="microbiome",
            wikipedia_title="Microbiota"),
    Concept("aging_senescence", "Cellular senescence / aging", "Genomics",
            openalex_search='"cellular senescence" OR "senolytic"',
            biorxiv_terms=["senescence", "senolytic"], nih_term="cellular senescence",
            wikipedia_title="Cellular_senescence"),

    # ---------------------------- AI / Machine Learning / CS ----------------------------
    Concept("deep_learning", "Deep learning", "AI/ML",
            openalex_search='"deep learning"',
            arxiv_query='abs:"deep learning"', nsf_query="deep learning",
            wikipedia_title="Deep_learning"),
    Concept("transformers", "Transformer architectures", "AI/ML",
            openalex_search='"transformer" AND ("attention" OR "neural network")',
            arxiv_query='abs:"transformer" AND cat:cs.LG',
            wikipedia_title="Transformer_(deep_learning_architecture)"),
    Concept("llms", "Large language models", "AI/ML",
            openalex_search='"large language model" OR "LLM"',
            arxiv_query='abs:"large language model"',
            wikipedia_title="Large_language_model"),
    Concept("diffusion_models", "Diffusion generative models", "AI/ML",
            openalex_search='"diffusion model" AND ("generative" OR "image")',
            arxiv_query='abs:"diffusion model"',
            wikipedia_title="Diffusion_model"),
    Concept("graph_neural_nets", "Graph neural networks", "AI/ML",
            openalex_search='"graph neural network"',
            arxiv_query='abs:"graph neural network"',
            wikipedia_title="Graph_neural_network"),
    Concept("reinforcement_learning", "Reinforcement learning", "AI/ML",
            openalex_search='"reinforcement learning"',
            arxiv_query='abs:"reinforcement learning"',
            wikipedia_title="Reinforcement_learning"),
    Concept("federated_learning", "Federated learning", "AI/ML",
            openalex_search='"federated learning"',
            arxiv_query='abs:"federated learning"',
            wikipedia_title="Federated_learning"),

    # ---------------------------- Materials science ----------------------------
    Concept("perovskite_solar", "Perovskite solar cells", "Materials",
            openalex_search='"perovskite solar cell"',
            arxiv_query='abs:"perovskite solar"', nsf_query="perovskite solar cell",
            wikipedia_title="Perovskite_solar_cell"),
    Concept("graphene_2d", "2D materials / graphene", "Materials",
            openalex_search='"graphene" OR "2D material"',
            arxiv_query='abs:"graphene"', nsf_query="graphene",
            wikipedia_title="Graphene"),
    Concept("mof", "Metal-organic frameworks", "Materials",
            openalex_search='"metal-organic framework"',
            arxiv_query='abs:"metal-organic framework"', nsf_query="metal organic framework",
            wikipedia_title="Metal%E2%80%93organic_framework"),
    Concept("solid_state_battery", "Solid-state batteries", "Materials",
            openalex_search='"solid-state battery" OR "solid state battery" OR "solid-state electrolyte"',
            arxiv_query='abs:"solid-state battery"', nsf_query="solid state battery",
            wikipedia_title="Solid-state_battery"),

    # ---------------------------- Climate / Energy ----------------------------
    Concept("carbon_capture", "Carbon capture", "Climate/Energy",
            openalex_search='"carbon capture" OR "CO2 capture"',
            arxiv_query=None, nsf_query="carbon capture",
            wikipedia_title="Carbon_capture_and_storage"),
    Concept("green_hydrogen", "Green hydrogen", "Climate/Energy",
            openalex_search='"green hydrogen" OR "hydrogen electrolysis"',
            nsf_query="green hydrogen", wikipedia_title="Hydrogen_economy"),
    Concept("climate_modeling", "Climate modeling", "Climate/Energy",
            openalex_search='"climate model" OR "earth system model"',
            arxiv_query='abs:"climate model"', nsf_query="climate model",
            wikipedia_title="Climate_model"),
    Concept("lithium_ion", "Lithium-ion batteries", "Climate/Energy",
            openalex_search='"lithium-ion battery" OR "lithium ion battery" OR "li-ion battery"',
            arxiv_query='abs:"lithium-ion battery"', nsf_query="lithium ion battery",
            wikipedia_title="Lithium-ion_battery"),

    # ---------------------------- Quantum ----------------------------
    Concept("quantum_computing", "Quantum computing", "Quantum",
            openalex_search='"quantum computing" OR "quantum computer"',
            arxiv_query='cat:quant-ph AND abs:"quantum computing"',
            nsf_query="quantum computing", wikipedia_title="Quantum_computing"),
    Concept("quantum_error_correction", "Quantum error correction", "Quantum",
            openalex_search='"quantum error correction"',
            arxiv_query='abs:"quantum error correction"',
            nsf_query="quantum error correction", wikipedia_title="Quantum_error_correction"),
    Concept("topological_insulators", "Topological insulators", "Quantum",
            openalex_search='"topological insulator"',
            arxiv_query='abs:"topological insulator"',
            nsf_query="topological insulator", wikipedia_title="Topological_insulator"),

    # ---------------------------- Physics / Astronomy ----------------------------
    Concept("gravitational_waves", "Gravitational waves", "Physics",
            openalex_search='"gravitational wave"',
            arxiv_query='cat:gr-qc AND abs:"gravitational wave"',
            nsf_query="gravitational wave", wikipedia_title="Gravitational_wave"),
    Concept("exoplanets", "Exoplanets", "Physics",
            openalex_search='"exoplanet"',
            arxiv_query='cat:astro-ph.EP AND abs:"exoplanet"',
            nsf_query="exoplanet", wikipedia_title="Exoplanet"),

    # ---------------------------- Metabolic / Other biomedical ----------------------------
    Concept("glp1", "GLP-1 / metabolic drugs", "Biomedical",
            openalex_search='"GLP-1" OR "glucagon-like peptide"',
            biorxiv_terms=["glp-1", "glucagon-like peptide"], nih_term="GLP-1",
            wikipedia_title="Glucagon-like_peptide-1"),
    Concept("gut_brain_axis", "Gut-brain axis", "Biomedical",
            openalex_search='"gut-brain axis" OR "gut brain axis"',
            biorxiv_terms=["gut-brain", "gut brain"], nih_term="gut brain axis",
            wikipedia_title="Gut%E2%80%93brain_axis"),
]


# --------------------------------------------------------------------------------------
# Convenience accessors
# --------------------------------------------------------------------------------------
def all_concepts() -> List[Concept]:
    return list(CONCEPTS)


def domains() -> List[str]:
    seen = []
    for c in CONCEPTS:
        if c.domain not in seen:
            seen.append(c.domain)
    return seen


def by_key() -> dict:
    return {c.key: c for c in CONCEPTS}


if __name__ == "__main__":
    print(f"{len(CONCEPTS)} concepts across {len(domains())} domains:")
    for d in domains():
        members = [c.name for c in CONCEPTS if c.domain == d]
        print(f"  {d:16s}: {', '.join(members)}")
