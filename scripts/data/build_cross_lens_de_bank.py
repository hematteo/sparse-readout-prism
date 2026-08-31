#!/usr/bin/env python3
"""Build the English-German cross-lens prompt bank and its cognate-exclusion report.

Writes ``data/cross_lens/cross_lens_prompts_en_de.json`` (90 cross-lens prompts in
seven families plus 12 controls) and the companion exclusion report, the inputs
of the second-language-pair cells of the cross-lens study
(``tab:app-cross-lens-extension`` rows "EN--DE, Jacobian" and "EN--DE, Jacobian,
n=300"; ``fig:cross-lens-extension``). The bank mirrors the EN-ZH bank with the
script confound removed: both languages are Latin-script, so the surface-language
call is lexical and the cognate filter below is load-bearing.

Construction rules, fixed before any lens was fitted:
  - Every decomposed surface (form_a, form_b, nulls) must be a single token
    under the Qwen tokenizer; the builder tries the context-appropriate spacing
    first (no leading space inside a quoted frame, a leading space otherwise),
    then the alternative, and drops the item if neither is a single token.
  - Cognate exclusion: after NFKD diacritic stripping, ss/eszett folding and
    casefolding, items with Levenshtein(form_a, form_b) <= 2 or identical forms
    are excluded. Exclusions are reported, not hidden (30 of 132 candidates in
    the shipped bank).
  - null_targets are language-matched unrelated nouns [de_null, en_null],
    assigned round-robin, skipping any null equal to either form.

Paper run::

  build_cross_lens_de_bank.py --tokenizer Qwen/Qwen3.5-9B \
      --out data/cross_lens/cross_lens_prompts_en_de.json \
      --report data/cross_lens/cross_lens_prompts_en_de_exclusion_report.json
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

# (group, concept, prompt, de_surface, en_surface, answer_lang)
# answer_lang: which side is the in-context gold continuation (form_a).
CANDIDATES = [
    # --- antonym_de: German few-shot antonym completion, German answer ---
    ("antonym_de", "big", 'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "klein" ist "', "groß", "big", "de"),
    (
        "antonym_de",
        "small",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "groß" ist "',
        "klein",
        "small",
        "de",
    ),
    (
        "antonym_de",
        "slow",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "schnell" ist "',
        "langsam",
        "slow",
        "de",
    ),
    (
        "antonym_de",
        "bright",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "dunkel" ist "',
        "hell",
        "bright",
        "de",
    ),
    (
        "antonym_de",
        "light",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "schwer" ist "',
        "leicht",
        "light",
        "de",
    ),
    (
        "antonym_de",
        "bad",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "gut" ist "',
        "schlecht",
        "bad",
        "de",
    ),
    (
        "antonym_de",
        "empty",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "voll" ist "',
        "leer",
        "empty",
        "de",
    ),
    (
        "antonym_de",
        "dry",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "nass" ist "',
        "trocken",
        "dry",
        "de",
    ),
    (
        "antonym_de",
        "weak",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "stark" ist "',
        "schwach",
        "weak",
        "de",
    ),
    (
        "antonym_de",
        "dark",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "hell" ist "',
        "dunkel",
        "dark",
        "de",
    ),
    ("antonym_de", "late", 'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "früh" ist "', "spät", "late", "de"),
    (
        "antonym_de",
        "cheap",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "teuer" ist "',
        "billig",
        "cheap",
        "de",
    ),
    ("antonym_de", "new", 'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "alt" ist "', "neu", "new", "de"),
    (
        "antonym_de",
        "quiet",
        'Das Gegenteil von "hoch" ist "tief". Das Gegenteil von "laut" ist "',
        "leise",
        "quiet",
        "de",
    ),
    # --- trans_de2en: German metalinguistic frame, English answer ---
    (
        "trans_de2en",
        "dog",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Hund" ist "',
        "Hund",
        "dog",
        "en",
    ),
    (
        "trans_de2en",
        "tree",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Baum" ist "',
        "Baum",
        "tree",
        "en",
    ),
    (
        "trans_de2en",
        "table",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Tisch" ist "',
        "Tisch",
        "table",
        "en",
    ),
    (
        "trans_de2en",
        "chair",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Stuhl" ist "',
        "Stuhl",
        "chair",
        "en",
    ),
    (
        "trans_de2en",
        "window",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Fenster" ist "',
        "Fenster",
        "window",
        "en",
    ),
    (
        "trans_de2en",
        "horse",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Pferd" ist "',
        "Pferd",
        "horse",
        "en",
    ),
    (
        "trans_de2en",
        "bird",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Vogel" ist "',
        "Vogel",
        "bird",
        "en",
    ),
    (
        "trans_de2en",
        "mountain",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Berg" ist "',
        "Berg",
        "mountain",
        "en",
    ),
    (
        "trans_de2en",
        "rain",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Regen" ist "',
        "Regen",
        "rain",
        "en",
    ),
    (
        "trans_de2en",
        "cheese",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Käse" ist "',
        "Käse",
        "cheese",
        "en",
    ),
    (
        "trans_de2en",
        "egg",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Ei" ist "',
        "Ei",
        "egg",
        "en",
    ),
    (
        "trans_de2en",
        "door",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Tür" ist "',
        "Tür",
        "door",
        "en",
    ),
    (
        "trans_de2en",
        "flower",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Blume" ist "',
        "Blume",
        "flower",
        "en",
    ),
    (
        "trans_de2en",
        "knife",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Messer" ist "',
        "Messer",
        "knife",
        "en",
    ),
    # --- trans_en2de: English metalinguistic frame, German answer ---
    (
        "trans_en2de",
        "dog",
        'The German word for "sky" is "Himmel". The German word for "dog" is "',
        "Hund",
        "dog",
        "de",
    ),
    (
        "trans_en2de",
        "tree",
        'The German word for "sky" is "Himmel". The German word for "tree" is "',
        "Baum",
        "tree",
        "de",
    ),
    (
        "trans_en2de",
        "table",
        'The German word for "sky" is "Himmel". The German word for "table" is "',
        "Tisch",
        "table",
        "de",
    ),
    (
        "trans_en2de",
        "chair",
        'The German word for "sky" is "Himmel". The German word for "chair" is "',
        "Stuhl",
        "chair",
        "de",
    ),
    (
        "trans_en2de",
        "window",
        'The German word for "sky" is "Himmel". The German word for "window" is "',
        "Fenster",
        "window",
        "de",
    ),
    (
        "trans_en2de",
        "horse",
        'The German word for "sky" is "Himmel". The German word for "horse" is "',
        "Pferd",
        "horse",
        "de",
    ),
    (
        "trans_en2de",
        "bird",
        'The German word for "sky" is "Himmel". The German word for "bird" is "',
        "Vogel",
        "bird",
        "de",
    ),
    (
        "trans_en2de",
        "mountain",
        'The German word for "sky" is "Himmel". The German word for "mountain" is "',
        "Berg",
        "mountain",
        "de",
    ),
    (
        "trans_en2de",
        "rain",
        'The German word for "sky" is "Himmel". The German word for "rain" is "',
        "Regen",
        "rain",
        "de",
    ),
    (
        "trans_en2de",
        "cheese",
        'The German word for "sky" is "Himmel". The German word for "cheese" is "',
        "Käse",
        "cheese",
        "de",
    ),
    ("trans_en2de", "egg", 'The German word for "sky" is "Himmel". The German word for "egg" is "', "Ei", "egg", "de"),
    (
        "trans_en2de",
        "door",
        'The German word for "sky" is "Himmel". The German word for "door" is "',
        "Tür",
        "door",
        "de",
    ),
    (
        "trans_en2de",
        "flower",
        'The German word for "sky" is "Himmel". The German word for "flower" is "',
        "Blume",
        "flower",
        "de",
    ),
    (
        "trans_en2de",
        "knife",
        'The German word for "sky" is "Himmel". The German word for "knife" is "',
        "Messer",
        "knife",
        "de",
    ),
    (
        "trans_de2en",
        "train",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Zug" ist "',
        "Zug",
        "train",
        "en",
    ),
    (
        "trans_de2en",
        "city",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Stadt" ist "',
        "Stadt",
        "city",
        "en",
    ),
    (
        "trans_de2en",
        "forest",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Wald" ist "',
        "Wald",
        "forest",
        "en",
    ),
    (
        "trans_de2en",
        "sea",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Meer" ist "',
        "Meer",
        "sea",
        "en",
    ),
    (
        "trans_de2en",
        "fire",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Feuer" ist "',
        "Feuer",
        "fire",
        "en",
    ),
    (
        "trans_de2en",
        "newspaper",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Zeitung" ist "',
        "Zeitung",
        "newspaper",
        "en",
    ),
    (
        "trans_de2en",
        "kitchen",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Küche" ist "',
        "Küche",
        "kitchen",
        "en",
    ),
    (
        "trans_de2en",
        "sister",
        'Das englische Wort für "Himmel" ist "sky". Das englische Wort für "Schwester" ist "',
        "Schwester",
        "sister",
        "en",
    ),
    (
        "trans_en2de",
        "train",
        'The German word for "sky" is "Himmel". The German word for "train" is "',
        "Zug",
        "train",
        "de",
    ),
    (
        "trans_en2de",
        "city",
        'The German word for "sky" is "Himmel". The German word for "city" is "',
        "Stadt",
        "city",
        "de",
    ),
    (
        "trans_en2de",
        "forest",
        'The German word for "sky" is "Himmel". The German word for "forest" is "',
        "Wald",
        "forest",
        "de",
    ),
    (
        "trans_en2de",
        "sea",
        'The German word for "sky" is "Himmel". The German word for "sea" is "',
        "Meer",
        "sea",
        "de",
    ),
    (
        "trans_en2de",
        "fire",
        'The German word for "sky" is "Himmel". The German word for "fire" is "',
        "Feuer",
        "fire",
        "de",
    ),
    (
        "trans_en2de",
        "newspaper",
        'The German word for "sky" is "Himmel". The German word for "newspaper" is "',
        "Zeitung",
        "newspaper",
        "de",
    ),
    (
        "trans_en2de",
        "kitchen",
        'The German word for "sky" is "Himmel". The German word for "kitchen" is "',
        "Küche",
        "kitchen",
        "de",
    ),
    (
        "trans_en2de",
        "sister",
        'The German word for "sky" is "Himmel". The German word for "sister" is "',
        "Schwester",
        "sister",
        "de",
    ),
    # --- cloze_de: German sentence completion, German noun answer ---
    ("cloze_de", "pen", "Zum Schreiben benutze ich einen", "Stift", "pen", "de"),
    ("cloze_de", "knife", "Ich schneide das Brot mit einem scharfen", "Messer", "knife", "de"),
    ("cloze_de", "station", "Der Zug hält pünktlich am", "Bahnhof", "station", "de"),
    ("cloze_de", "door", "Ich klopfe dreimal und öffne dann die", "Tür", "door", "de"),
    ("cloze_de", "book", "Vor dem Schlafen liest das Kind noch ein", "Buch", "book", "de"),
    ("cloze_de", "tree", "Der Vogel sitzt hoch oben auf dem", "Baum", "tree", "de"),
    ("cloze_de", "snow", "Im Winter fällt auf den Bergen viel", "Schnee", "snow", "de"),
    ("cloze_de", "coffee", "Zum Frühstück trinke ich immer eine Tasse", "Kaffee", "coffee", "de"),
    ("cloze_de", "key", "Ich schließe die Wohnung ab mit dem", "Schlüssel", "key", "de"),
    ("cloze_de", "horse", "Auf dem Bauernhof reitet das Mädchen ein", "Pferd", "horse", "de"),
    ("cloze_de", "cheese", "Zur Brotzeit essen wir Brot mit", "Käse", "cheese", "de"),
    ("cloze_de", "flower", "Zum Geburtstag schenke ich ihr eine", "Blume", "flower", "de"),
    ("cloze_de", "school", "Nach den Ferien gehen die Kinder wieder in die", "Schule", "school", "de"),
    ("cloze_de", "church", "Sonntags läuten die Glocken der alten", "Kirche", "church", "de"),
    ("cloze_de", "train", "Wir fahren in den Urlaub bequem mit dem", "Zug", "train", "de"),
    ("cloze_de", "forest", "Am Wochenende wandern wir stundenlang durch den", "Wald", "forest", "de"),
    ("cloze_de", "ship", "Der Kapitän steuert das große", "Schiff", "ship", "de"),
    ("cloze_de", "money", "Für die Fahrkarte brauche ich noch etwas", "Geld", "money", "de"),
    # --- cloze_en: English sentence completion, English answer ---
    ("cloze_en", "key", "I unlock the front door with my", "Schlüssel", "key", "en"),
    ("cloze_en", "horse", "At the farm the girl rides a brown", "Pferd", "horse", "en"),
    ("cloze_en", "window", "She waved at us through the kitchen", "Fenster", "window", "en"),
    ("cloze_en", "tree", "The bird built its nest high up in the", "Baum", "tree", "en"),
    ("cloze_en", "table", "He put the plates down on the wooden", "Tisch", "table", "en"),
    ("cloze_en", "dog", "The postman was chased down the street by a barking", "Hund", "dog", "en"),
    ("cloze_en", "rain", "Take an umbrella, the forecast says heavy", "Regen", "rain", "en"),
    ("cloze_en", "cheese", "The mouse in the cartoon always steals the", "Käse", "cheese", "en"),
    ("cloze_en", "egg", "For breakfast she boiled herself an", "Ei", "egg", "en"),
    ("cloze_en", "mountain", "They spent all summer climbing the highest", "Berg", "mountain", "en"),
    ("cloze_en", "knife", "He spread the butter with a blunt", "Messer", "knife", "en"),
    ("cloze_en", "bird", "High above the field we watched a small", "Vogel", "bird", "en"),
    ("cloze_en", "train", "We traveled to the capital on a high-speed", "Zug", "train", "en"),
    ("cloze_en", "money", "He paid for the tickets with borrowed", "Geld", "money", "en"),
    ("cloze_en", "voice", "The choir admired her beautiful singing", "Stimme", "voice", "en"),
    ("cloze_en", "newspaper", "Every morning my grandfather reads the", "Zeitung", "newspaper", "en"),
    ("cloze_en", "war", "After years of peace the country slid into", "Krieg", "war", "en"),
    ("cloze_en", "city", "From the hilltop you can see the whole", "Stadt", "city", "en"),
    # --- fact_de: German general knowledge, German common-noun answer ---
    ("fact_de", "lion", "Der König der Tiere ist der", "Löwe", "lion", "de"),
    ("fact_de", "dog", "Der beste Freund des Menschen ist der", "Hund", "dog", "de"),
    ("fact_de", "bee", "Honig wird gemacht von der fleißigen", "Biene", "bee", "de"),
    ("fact_de", "moon", "Nachts sieht man am Himmel die Sterne und den", "Mond", "moon", "de"),
    ("fact_de", "snow", "Weiß und kalt fällt im Winter der", "Schnee", "snow", "de"),
    ("fact_de", "egg", "Die Henne legt jeden Morgen ein", "Ei", "egg", "de"),
    ("fact_de", "horse", "Vor die Kutsche spannt man ein", "Pferd", "horse", "de"),
    ("fact_de", "cheese", "Aus Milch macht man Butter und", "Käse", "cheese", "de"),
    ("fact_de", "week", "Sieben Tage sind eine", "Woche", "week", "de"),
    ("fact_de", "year", "Zwölf Monate sind ein", "Jahr", "year", "de"),
    ("fact_de", "tooth", "Der Zahnarzt zieht den kranken", "Zahn", "tooth", "de"),
    ("fact_de", "island", "Ein Stück Land mitten im Meer nennt man eine", "Insel", "island", "de"),
    ("fact_de", "church", "Am Sonntag gehen viele Menschen in die", "Kirche", "church", "de"),
    # --- exemplar_de: German category exemplars, German answer ---
    ("exemplar_de", "dog", "Ein Beispiel für ein Haustier ist der", "Hund", "dog", "de"),
    ("exemplar_de", "table", "Ein Beispiel für ein Möbelstück ist der", "Tisch", "table", "de"),
    ("exemplar_de", "juice", "Ein Beispiel für ein Getränk ist der", "Saft", "juice", "de"),
    ("exemplar_de", "doctor", "Ein Beispiel für einen Beruf ist der", "Arzt", "doctor", "de"),
    ("exemplar_de", "teacher", "Ein Beispiel für einen Beruf ist der", "Lehrer", "teacher", "de"),
    ("exemplar_de", "coat", "Ein Beispiel für ein Kleidungsstück ist der", "Mantel", "coat", "de"),
    ("exemplar_de", "shirt", "Ein Beispiel für ein Kleidungsstück ist das", "Hemd", "shirt", "de"),
    ("exemplar_de", "bird", "Ein Beispiel für ein Tier mit Flügeln ist der", "Vogel", "bird", "de"),
    ("exemplar_de", "head", "Ein Beispiel für ein Körperteil ist der", "Kopf", "head", "de"),
    ("exemplar_de", "sea", "Ein Beispiel für ein Gewässer ist das", "Meer", "sea", "de"),
    ("exemplar_de", "train", "Ein Beispiel für ein Verkehrsmittel ist der", "Zug", "train", "de"),
    ("exemplar_de", "church", "Ein Beispiel für ein Gebäude ist die", "Kirche", "church", "de"),
    ("exemplar_de", "sister", "Ein Beispiel für ein Familienmitglied ist die", "Schwester", "sister", "de"),
    # --- controls: no language flip expected ---
    ("ctrl_digit", "seven", "3 + 4 =", "7", "7", "de"),
    ("ctrl_digit", "nine", "5 + 4 =", "9", "9", "de"),
    ("ctrl_digit", "eight", "6 + 2 =", "8", "8", "de"),
    ("ctrl_digit", "six", "2 + 4 =", "6", "6", "de"),
    ("ctrl_digit", "five", "1 + 4 =", "5", "5", "de"),
    ("ctrl_digit", "four", "1 + 3 =", "4", "4", "de"),
    ("ctrl_propn", "paris", "Die Hauptstadt von Frankreich heißt", "Paris", "Paris", "de"),
    ("ctrl_propn", "berlin", "Die Hauptstadt von Deutschland heißt", "Berlin", "Berlin", "de"),
    ("ctrl_propn", "london", "Die Hauptstadt von England heißt", "London", "London", "de"),
    ("ctrl_propn", "madrid", "Die Hauptstadt von Spanien heißt", "Madrid", "Madrid", "de"),
    ("ctrl_propn", "oslo", "Die Hauptstadt von Norwegen heißt", "Oslo", "Oslo", "de"),
    ("ctrl_propn", "lima", "Die Hauptstadt von Peru heißt", "Lima", "Lima", "de"),
]

DE_NULLS = ["Salz", "Wand", "Schuh", "Tasse", "Nadel", "Karte", "Seife", "Gabel"]
EN_NULLS = ["salt", "wall", "shoe", "cup", "needle", "ticket", "soap", "fork"]

CROSS_GROUPS = {
    "antonym_de",
    "trans_de2en",
    "trans_en2de",
    "cloze_de",
    "cloze_en",
    "fact_de",
    "exemplar_de",
}

NOTE = (
    "Cross-lens prompt bank EN-DE. Same schema and metrics as "
    "cross_lens_prompts_en_zh.json; both languages Latin-script, so the "
    "surface-language call is lexical and cognates (normalized Levenshtein "
    "<= 2) are excluded by construction. form_a = in-context answer form, "
    "form_b = other-language form; null_targets = [de, en] language-matched "
    "unrelated tokens. Built by scripts/data/build_cross_lens_de_bank.py; "
    "exclusions in the companion report cross_lens_prompts_en_de_exclusion_report.json."
)


def norm(s: str) -> str:
    s = s.strip().casefold().replace("ß", "ss")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def single_token(tok, surface: str, prefer_space: bool) -> str | None:
    """Return the single-token spacing variant of `surface`, or None."""
    variants = [" " + surface, surface] if prefer_space else [surface, " " + surface]
    for v in variants:
        if len(tok.encode(v, add_special_tokens=False)) == 1:
            return v
    return None


def build_bank(tok) -> tuple[list[dict], list[dict]]:
    """Apply the construction rules to CANDIDATES; returns (prompts, excluded)."""
    prompts, excluded = [], []
    counters: dict[str, int] = {}
    null_i = 0
    for group, concept, prompt, de, en, answer_lang in CANDIDATES:
        counters[group] = counters.get(group, 0) + 1
        pid = f"{group}_{counters[group]:02d}"
        is_ctrl = group.startswith("ctrl_")

        if not is_ctrl and (norm(de) == norm(en) or levenshtein(norm(de), norm(en)) <= 2):
            excluded.append(
                {
                    "id": pid,
                    "concept": concept,
                    "de": de,
                    "en": en,
                    "reason": f"cognate (lev={levenshtein(norm(de), norm(en))})",
                }
            )
            continue

        # In-context continuation: quoted frames take no leading space, plain
        # sentence frames do. The answer-language surface is form_a.
        quoted = prompt.rstrip().endswith('"')
        de_tok = single_token(tok, de, prefer_space=not quoted if answer_lang == "de" else True)
        en_tok = single_token(tok, en, prefer_space=not quoted if answer_lang == "en" else True)
        if de_tok is None or en_tok is None:
            excluded.append(
                {
                    "id": pid,
                    "concept": concept,
                    "de": de,
                    "en": en,
                    "reason": "not single-token: "
                    + ("de " if de_tok is None else "")
                    + ("en" if en_tok is None else ""),
                }
            )
            continue

        form_a, form_b = (de_tok, en_tok) if answer_lang == "de" else (en_tok, de_tok)

        # Language-matched nulls, skipping collisions with either form.
        nulls = []
        for pool, prefer in ((DE_NULLS, True), (EN_NULLS, True)):
            for step in range(len(pool)):
                cand = pool[(null_i + step) % len(pool)]
                if norm(cand) in (norm(de), norm(en)):
                    continue
                cand_tok = single_token(tok, cand, prefer_space=prefer)
                if cand_tok is not None:
                    nulls.append(cand_tok)
                    break
        null_i += 1
        if len(nulls) < 2:
            excluded.append({"id": pid, "concept": concept, "de": de, "en": en, "reason": "no single-token null pair"})
            continue

        prompts.append(
            {
                "id": pid,
                "group": group,
                "concept": concept,
                "prompt": prompt,
                "form_a": form_a,
                "form_b": form_b,
                "lang_a": answer_lang,
                "lang_b": "en" if answer_lang == "de" else "de",
                "targets": [form_a, form_b, *nulls],
                "null_targets": nulls,
                "answer": form_a.strip(),
            }
        )
    return prompts, excluded


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--out", required=True, help="output bank JSON")
    ap.add_argument("--report", required=True, help="output exclusion report JSON")
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prompts, excluded = build_bank(tok)

    n_cross = sum(1 for p in prompts if p["group"] in CROSS_GROUPS)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"note": NOTE, "prompts": prompts}, ensure_ascii=False, indent=1))
    report = {
        "tokenizer": args.tokenizer,
        "n_candidates": len(CANDIDATES),
        "n_kept": len(prompts),
        "n_cross": n_cross,
        "n_ctrl": len(prompts) - n_cross,
        "per_group": {g: sum(1 for p in prompts if p["group"] == g) for g in sorted({p["group"] for p in prompts})},
        "excluded": excluded,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(
        f"[bank] kept {len(prompts)} ({n_cross} cross + {len(prompts) - n_cross} ctrl), "
        f"excluded {len(excluded)} of {len(CANDIDATES)} candidates"
    )
    for e in excluded:
        print(f"  - {e['id']} {e['de']}/{e['en']}: {e['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
