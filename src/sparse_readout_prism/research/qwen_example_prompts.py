"""Static Qwen contrast-prompt example banks used by the paper figures.

Extracted from ``scripts/figures/compute_sae_paper_examples.py`` so the prompt
constants are importable by the analysis scripts as well as that entry point.
Each entry is a ``case_id``/``title``/``prompt``/``target_a``/
``target_b``/``claim`` dict; ``EXAMPLE_SETS`` maps named sets to these lists.

Only the sets behind shipped paper artifacts are kept:
``section43_polysemy_margin`` backs the main case-study four-panel figure
(``fig:selected-readout-prism-examples``); ``section43_margin_appendix2`` is
its bark/bass subset rerun for the Appendix L margin case studies.
"""

from __future__ import annotations

__all__ = [
    "SECTION43_POLYSEMY_MARGIN_EXAMPLES",
    "SECTION43_MARGIN_APPENDIX2",
    "EXAMPLE_SETS",
]


SECTION43_POLYSEMY_MARGIN_EXAMPLES = [
    {
        "case_id": "bug_insect_error",
        "title": "Insect context",
        "prompt": "The child watched a tiny crawling",
        "target_a": " bug",
        "target_b": " error",
        "claim": "insect reading over software-error reading",
    },
    {
        "case_id": "bug_software_insect",
        "title": "Software context",
        "prompt": "The programmer reproduced the crash and filed a",
        "target_a": " bug",
        "target_b": " insect",
        "claim": "software bug reading over animal/insect reading",
    },
    {
        "case_id": "bark_tree_dog",
        "title": "Tree context",
        "prompt": "The old oak had rough",
        "target_a": " bark",
        "target_b": " dog",
        "claim": "tree-surface reading over animal reading",
    },
    {
        "case_id": "bass_music_trout",
        "title": "Music context",
        "prompt": "The song sounded thin, so the producer added more",
        "target_a": " bass",
        "target_b": " trout",
        "claim": "music reading over fish reading",
    },
]


SECTION43_MARGIN_APPENDIX2 = [
    SECTION43_POLYSEMY_MARGIN_EXAMPLES[2],
    SECTION43_POLYSEMY_MARGIN_EXAMPLES[3],
]


EXAMPLE_SETS = {
    "section43_margin_appendix2": SECTION43_MARGIN_APPENDIX2,
    "section43_polysemy_margin": SECTION43_POLYSEMY_MARGIN_EXAMPLES,
}
