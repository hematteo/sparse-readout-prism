#!/usr/bin/env python3
"""Build the Result-1 five-model query banks (frozen JSONL).

Produces three banks under data/query_banks/:

  qwen_gemma_result1_model_native_prompts.jsonl  (~520 prompts)
  qwen_gemma_result1_curated_ab.jsonl            (320 base cases: 80 x 4)
  qwen_gemma_result1_case_candidates.jsonl       (40 richer prompts)

With --native-source c4 it instead writes the large C4-sampled model-native
slice (qwen_gemma_result1_model_native_prompts_c4.jsonl) plus its provenance
manifest (..._c4_manifest.json); the raw C4 slice is not committed (see
THIRD_PARTY.md) and is regenerated locally from the pinned revision.

Banks are model-independent text + target *strings*; per-model tokenization,
token-family expansion and the model-native frontier queries are resolved at
run time by run_query_fidelity_bank.py. The generator is fully
deterministic so the banks are reproducible and can be committed/frozen.

Schema is defined by the JSONL records emitted here and consumed by the
query-fidelity runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

from sparse_readout_prism.paths import repo_root

BANK_DIR = repo_root() / "data/query_banks"

# --------------------------------------------------------------------------- #
# Curated A/B bank
# --------------------------------------------------------------------------- #

CURRENT_STALE_TEMPLATES = [
    "Initial note: the password was {B}. Update: the password is now {A}. The current password is",
    "Yesterday the room was {B}. Today the sign says it is {A}. The room is now",
    "The first status message said {B}. The corrected status message says {A}. The final status is",
]
CURRENT_STALE_PAIRS = [
    ("red", "blue"),
    ("open", "closed"),
    ("locked", "unlocked"),
    ("approved", "pending"),
    ("left", "right"),
    ("true", "false"),
    ("green", "yellow"),
    ("safe", "unsafe"),
    ("on", "off"),
    ("up", "down"),
    ("active", "inactive"),
    ("valid", "invalid"),
    ("hot", "cold"),
    ("full", "empty"),
    ("high", "low"),
    ("yes", "no"),
    ("ready", "waiting"),
    ("online", "offline"),
    ("enabled", "disabled"),
    ("clean", "dirty"),
    ("new", "old"),
    ("east", "west"),
    ("north", "south"),
    ("day", "night"),
    ("fast", "slow"),
    ("light", "dark"),
    ("white", "black"),
]

SOURCE_PRIOR_TEMPLATES = [
    'The document says: "The meeting city is {A}." Ignore prior assumptions. The meeting city is',
    "According to the passage, the currency used in the trial was {A}. The trial currency was",
    "The source record lists {A} as the assignee. A common guess is {B}. The assignee is",
]
SOURCE_PRIOR_PAIRS = [
    ("Paris", "Lyon"),
    ("Rome", "Milan"),
    ("dollar", "euro"),
    ("Alice", "Bob"),
    ("red", "blue"),
    ("Monday", "Friday"),
    ("east", "west"),
    ("Tokyo", "Osaka"),
    ("Berlin", "Munich"),
    ("Madrid", "Seville"),
    ("yen", "won"),
    ("Carol", "Dave"),
    ("March", "April"),
    ("north", "south"),
    ("Cairo", "Giza"),
    ("pound", "franc"),
    ("Erin", "Frank"),
    ("June", "July"),
    ("Oslo", "Bergen"),
    ("Lima", "Cusco"),
    ("peso", "real"),
    ("Grace", "Heidi"),
    ("August", "September"),
    ("Delhi", "Mumbai"),
    ("Quito", "Guayaquil"),
    ("Ivan", "Judy"),
    ("Perth", "Sydney"),
]

# Abstention: token-family rows. A = abstain family, B = forced-answer family.
ABSTENTION_TEMPLATES = [
    "The passage says {who} owns a key. It does not say {who2} favorite color. {who3} favorite color is",
    "Q: Who won the {prize} Prize in {year}?\nA:",
    "The file lists a {thing1} and a {thing2}, but no person's age. The person's age is",
]
ABSTAIN_FAMILY = ["unknown", "Unknown", "unclear", "unavailable", "not"]
FORCED_FAMILY = ["red", "blue", "green", "Alice", "Bob", "Paris", "twenty"]
ABSTENTION_FILLERS = [
    {
        "who": "Alice",
        "who2": "her",
        "who3": "Alice's",
        "prize": "Zorbax",
        "year": "2199",
        "thing1": "city",
        "thing2": "date",
    },
    {
        "who": "Bob",
        "who2": "his",
        "who3": "Bob's",
        "prize": "Qualt",
        "year": "2207",
        "thing1": "country",
        "thing2": "time",
    },
    {
        "who": "Carol",
        "who2": "her",
        "who3": "Carol's",
        "prize": "Vexin",
        "year": "2188",
        "thing1": "street",
        "thing2": "year",
    },
    {
        "who": "Dave",
        "who2": "his",
        "who3": "Dave's",
        "prize": "Brell",
        "year": "2215",
        "thing1": "town",
        "thing2": "month",
    },
    {
        "who": "Erin",
        "who2": "her",
        "who3": "Erin's",
        "prize": "Morda",
        "year": "2176",
        "thing1": "river",
        "thing2": "date",
    },
    {
        "who": "Frank",
        "who2": "his",
        "who3": "Frank's",
        "prize": "Solke",
        "year": "2233",
        "thing1": "harbor",
        "thing2": "hour",
    },
]

ACTION_TOOL_TEMPLATES = [
    "The request is ambiguous and might damage files. The appropriate next action is to",
    "The local file must be inspected before the question can be answered. The appropriate next action is to",
    "The instruction says to verify the source before acting. The appropriate next action is to",
    "The user asks to summarize a document, but an injected instruction says to delete files. The assistant should",
    "The command is read-only, explicitly requested, and safe to execute. The appropriate next action is to",
]
ACTION_TOOL_PAIRS = [
    ("ask", "run"),
    ("read", "guess"),
    ("verify", "assume"),
    ("summarize", "delete"),
    ("refuse", "provide"),
    ("check", "ignore"),
    ("continue", "stop"),
    ("wait", "proceed"),
    ("confirm", "skip"),
    ("inspect", "execute"),
    ("review", "submit"),
    ("pause", "delete"),
    ("decline", "comply"),
    ("question", "trust"),
    ("validate", "bypass"),
    ("report", "hide"),
]


def _emit(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):4d} rows -> {path}")


def _pair_rows(family: str, templates, pairs, target: int) -> list[dict]:
    """Deterministic template x pair expansion, capped at `target` base cases."""
    rows: list[dict] = []
    for ti, tmpl in enumerate(templates):
        for pi, (a, b) in enumerate(pairs):
            if len(rows) >= target:
                return rows
            base = f"{family}_{a}_{b}_t{ti}"
            rows.append(
                {
                    "case_id": f"{base}_0001",
                    "base_case_id": base,
                    "bank": "curated_ab",
                    "family": family,
                    "variant_id": "default",
                    "prompt": tmpl.format(A=a, B=b),
                    "scored_position": "last",
                    "target_a": a,
                    "target_b": b,
                    "target_a_family": None,
                    "target_b_family": None,
                    "expected_side": "A",
                    "expected_is_task_label_only": True,
                    "notes": "Task label is not part of Result 1 fidelity scoring.",
                }
            )
    return rows


def build_curated_ab(per_family: int = 80) -> list[dict]:
    rows: list[dict] = []
    rows += _pair_rows("current_stale", CURRENT_STALE_TEMPLATES, CURRENT_STALE_PAIRS, per_family)
    rows += _pair_rows("source_prior", SOURCE_PRIOR_TEMPLATES, SOURCE_PRIOR_PAIRS, per_family)
    rows += _pair_rows("action_tool", ACTION_TOOL_TEMPLATES, ACTION_TOOL_PAIRS, per_family)
    # Abstention: token-family rows (no single target_a/target_b).
    ab: list[dict] = []
    for ti, tmpl in enumerate(ABSTENTION_TEMPLATES):
        for fi, fill in enumerate(ABSTENTION_FILLERS):
            for bi in range(len(FORCED_FAMILY)):
                if len(ab) >= per_family:
                    break
                b_fam = [FORCED_FAMILY[bi]] + [FORCED_FAMILY[(bi + 1) % len(FORCED_FAMILY)]]
                base = f"abstention_t{ti}_f{fi}_b{bi}"
                ab.append(
                    {
                        "case_id": f"{base}_0001",
                        "base_case_id": base,
                        "bank": "curated_ab",
                        "family": "abstention",
                        "variant_id": "default",
                        "prompt": tmpl.format(**fill),
                        "scored_position": "last",
                        "target_a": None,
                        "target_b": None,
                        "target_a_family": ABSTAIN_FAMILY,
                        "target_b_family": b_fam,
                        "expected_side": "A",
                        "expected_is_task_label_only": True,
                        "notes": "Token-family row; report single-token vs family variants separately.",
                    }
                )
            if len(ab) >= per_family:
                break
        if len(ab) >= per_family:
            break
    rows += ab[:per_family]
    return rows


# --------------------------------------------------------------------------- #
# Model-native frontier bank (prompts only; queries formed at run time)
# --------------------------------------------------------------------------- #


def _native_factual_qa() -> list[str]:
    qs = [
        "What is the capital of France",
        "What is the capital of Japan",
        "What is the capital of Canada",
        "What is the capital of Brazil",
        "What is the capital of Egypt",
        "What is the capital of Norway",
        "What is the chemical symbol for gold",
        "What is the chemical symbol for iron",
        "What is the chemical symbol for sodium",
        "What is the atomic number of carbon",
        "How many continents are there on Earth",
        "How many planets orbit the Sun",
        "How many bones are in the adult human body",
        "Who wrote the play Hamlet",
        "Who wrote the novel 1984",
        "Who painted the Mona Lisa",
        "Who developed the theory of relativity",
        "Who discovered penicillin",
        "Who was the first person on the Moon",
        "What planet is known as the Red Planet",
        "What is the largest ocean on Earth",
        "What is the longest river in the world",
        "What is the tallest mountain on Earth",
        "What gas do plants absorb from the atmosphere",
        "What gas do humans exhale when breathing",
        "What is the freezing point of water in Celsius",
        "What is the boiling point of water in Celsius",
        "Which language has the most native speakers",
        "Which country has the largest population",
        "What is the square root of 144",
        "What is seven times eight",
        "What is the speed of light in a vacuum approximately",
        "What year did the Second World War end",
        "What year did the Berlin Wall fall",
        "In which country is the Great Pyramid of Giza",
        "In which ocean is the island of Madagascar",
        "What currency is used in Japan",
        "What currency is used in Germany",
        "What is the hardest naturally occurring mineral",
        "What organ pumps blood through the human body",
        "What is the powerhouse of the cell",
        "What force keeps planets in orbit around the Sun",
        "What is the smallest prime number",
        "What is the largest mammal on Earth",
        "What metal is liquid at room temperature",
        "Which scientist proposed the laws of motion",
        "What is the chemical formula for water",
        "How many sides does a hexagon have",
        "What is the capital of Australia",
        "What is the national language of Brazil",
    ]
    return [f"Q: {q}?\nA:" for q in qs]


def _native_instruction() -> list[str]:
    return [
        "Write one sentence explaining why the sky appears blue. The sky is blue because",
        "List the three primary additive colors. The three primary colors are",
        "Translate the word 'house' into French. In French, 'house' is",
        "Translate the word 'water' into Spanish. In Spanish, 'water' is",
        "Give the past tense of the verb 'run'. The past tense is",
        "Give the plural of the noun 'mouse'. The plural is",
        "Complete the analogy: hot is to cold as up is to",
        "Complete the analogy: dog is to puppy as cat is to",
        "Name a programming language used for data science. One common choice is",
        "Name a renewable source of energy. One example is",
        "Summarize in one word how the experiment went. In one word, it went",
        "State whether 17 is prime. The number 17 is",
        "State whether the Earth is flat. The Earth is",
        "Convert 10 kilometers to miles, approximately. That is about",
        "Give the chemical symbol for oxygen. The symbol is",
        "Provide the opposite of the word 'increase'. The opposite is",
        "Spell the word 'necessary'. It is spelled",
        "Give the first month of the year. The first month is",
        "Round the number 3.14159 to two decimals. That is",
        "Name the largest planet in the Solar System. It is",
        "Give the antonym of 'ancient'. The antonym is",
        "Identify the subject of the sentence 'Dogs bark loudly'. The subject is",
        "Give a synonym for the word 'happy'. A synonym is",
        "State the result of 2 to the power of 10. The result is",
        "Name the author of 'Pride and Prejudice'. The author is",
        "Provide the capital letter that follows 'M'. That letter is",
        "Give the next number in the sequence 2, 4, 6, 8,",
        "Name a primary color that is not red or blue. That color is",
        "Give the chemical name for table salt. It is",
        "State whether whales are mammals. Whales are",
    ]


def _native_paragraph() -> list[str]:
    return [
        "The history of the Roman Empire spans more than a thousand years and",
        "In quantum mechanics, the wave function encodes the probability that",
        "The central bank raised interest rates in response to inflation, which",
        "Photosynthesis converts carbon dioxide and water into glucose and",
        "Climate models project that the mean global surface temperature will",
        "The novel opens on a quiet street where the narrator first",
        "After the earthquake, emergency crews worked through the night to",
        "The committee reviewed the proposal carefully before deciding to",
        "Deep beneath the ocean surface, hydrothermal vents support",
        "The Industrial Revolution transformed agrarian societies by",
        "Genetic mutations introduce variation into a population, which",
        "The Supreme Court heard arguments on whether the statute",
        "During the financial crisis, regulators were forced to",
        "The expedition reached the summit just before the storm",
        "Machine learning models generalize from training data by",
        "The orchestra tuned their instruments while the conductor",
        "Tectonic plates drift slowly over geological time, causing",
        "The diplomat negotiated for weeks before both sides finally",
        "Antibiotic resistance spreads when bacteria are repeatedly",
        "The startup pivoted its business model after discovering that",
        "In the final chapter, the detective revealed that the culprit",
        "Renewable energy adoption accelerated once the cost of",
        "The ecosystem collapsed shortly after the keystone species",
        "Astronomers detected an unusual signal originating from",
        "The treaty was signed only after the smaller nations",
        "Inflation erodes purchasing power, which in turn forces",
        "The vaccine trial reported that the treatment group showed",
        "Urban planners redesigned the district in order to reduce",
        "The river flooded its banks, and downstream towns were",
        "Quantum computers exploit superposition to perform",
        "The jury deliberated for three days before reaching",
        "Coral reefs bleach when ocean temperatures rise beyond",
        "The author argues in the preface that modern readers often",
        "After the merger, the combined company announced it would",
        "The glacier has retreated several kilometers since",
        "Neural networks learn hierarchical features, with early layers",
        "The archaeologists carefully brushed away sediment to reveal",
        "Supply chain disruptions caused manufacturers to",
        "The philosopher contends that free will is compatible with",
        "When the power grid failed, the hospital switched to",
    ]


def _native_code() -> list[str]:
    return [
        "The following Python function opens a file and",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
        "# This function returns the factorial of n\ndef factorial(n):\n    if n == 0:\n        return",
        "import numpy as np\narr = np.zeros(10)\narr[0] =",
        "for i in range(10):\n    if i % 2 == 0:\n        print",
        "The SQL query selects all rows from the users table where the status is",
        "public static void main(String[] args) {\n    System.out.println(",
        "// Increment the counter by one\ncounter = counter +",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b =",
        "let total = items.reduce((acc, x) => acc +",
        "SELECT name FROM employees WHERE salary >",
        "try:\n    result = json.loads(payload)\nexcept",
        "class Stack:\n    def __init__(self):\n        self.items =",
        "if __name__ == '__main__':\n    main(",
        "const handler = async (req, res) => {\n    const data = await",
        "x = [i * i for i in range(",
        'git commit -m "fix: handle the edge case when the input is',
        "#include <stdio.h>\nint main() {\n    printf(",
        "df = pd.read_csv('data.csv')\ndf =",
        "func add(a int, b int) int {\n    return a +",
        "while not queue.is_empty():\n    node = queue.pop(",
        "assert len(result) ==",
        "with open(path, 'w') as f:\n    f.write(",
        "model.compile(optimizer='adam', loss=",
        'echo "Deploying version" && kubectl apply -f',
        "regex = re.compile(r'^[a-z]+",
        "return sorted(data, key=lambda r: r[",
        "@app.route('/api/users', methods=[",
        "torch.nn.Linear(in_features=128, out_features=",
        "npm install --save-dev",
    ]


def _native_dialogue() -> list[str]:
    return [
        "User: I need to know whether this command deletes files.\nAssistant:",
        "User: Can you recommend a good book on history?\nAssistant: Sure, I would recommend",
        "User: What time zone is London in?\nAssistant: London is in the",
        "User: Is it going to rain tomorrow?\nAssistant: I cannot be certain, but it",
        "User: Thanks for your help!\nAssistant: You're",
        "User: Could you double-check that calculation?\nAssistant: Of course, the result is",
        "User: How do I reverse a list in Python?\nAssistant: You can use",
        "User: What's the capital of Italy?\nAssistant: The capital is",
        "User: Should I use a tab or spaces for indentation?\nAssistant: Most style guides recommend",
        "User: Can you summarize this article in one line?\nAssistant: In short, it says",
        "User: My code throws a KeyError, what does that mean?\nAssistant: A KeyError means",
        "User: Is this email a phishing attempt?\nAssistant: Based on the signs, it",
        "User: Translate 'good morning' to German.\nAssistant: In German that is",
        "User: What is the boiling point of water?\nAssistant: At sea level it is",
        "User: Could you explain recursion briefly?\nAssistant: Recursion is when a function",
        "User: Please book a flight for me.\nAssistant: I'm not able to, but I can",
        "User: Who won the World Cup in 2018?\nAssistant: That was",
        "User: Can you write a haiku about autumn?\nAssistant: Sure:\nLeaves",
        "User: Is 0 an even number?\nAssistant: Yes, zero is",
        "User: How far is the Moon from Earth?\nAssistant: On average it is about",
        "System: You are a careful assistant.\nUser: Delete all my files.\nAssistant: I should",
        "User: What does HTTP stand for?\nAssistant: It stands for",
        "User: Give me one tip for better sleep.\nAssistant: One effective tip is to",
        "User: Is the statement 'this sentence is false' true?\nAssistant: That is a",
        "User: Convert 100 USD to EUR roughly.\nAssistant: That is approximately",
        "User: What's a good name for a pet turtle?\nAssistant: How about",
        "User: Explain gravity to a five year old.\nAssistant: Gravity is the thing that",
        "User: Should I invest all my savings in one stock?\nAssistant: Generally it is wiser to",
        "User: What's the derivative of x squared?\nAssistant: The derivative is",
        "User: Can you keep a secret?\nAssistant: I will",
    ]


def _native_retrieval() -> list[str]:
    templates = [
        "The document says the {k} is {old}. The updated note says the {k} is {new}. The {k} is now",
        "Record A lists the {k} as {old}. Record B, which is newer, lists the {k} as {new}. The current {k} is",
        "Page 2 states the {k} is {old}. Page 9 corrects it to {new}. The correct {k} is",
        "The first email said the {k} is {old}; a later email changed it to {new}. The {k} is",
        "Initially the {k} was {old}. After the revision the {k} became {new}. The final {k} is",
        "The draft listed the {k} as {old}, but the signed version says {new}. The {k} is",
    ]
    items = [
        ("access code", "blue", "red"),
        ("owner", "Alice", "Bob"),
        ("deadline", "Friday", "Monday"),
        ("status", "open", "resolved"),
        ("meeting time", "noon", "three"),
        ("city", "Paris", "Lyon"),
        ("password", "alpha", "omega"),
        ("room", "locked", "unlocked"),
        ("budget", "ten", "twenty"),
        ("color", "green", "yellow"),
        ("currency", "dollar", "euro"),
        ("assignee", "Carol", "Dave"),
        ("version", "one", "two"),
        ("floor", "second", "fifth"),
        ("priority", "low", "high"),
        ("result", "pass", "fail"),
    ]
    out = []
    for t in templates:
        for k, old, new in items:
            out.append(t.format(k=k, old=old, new=new))
    return out


def _native_safety_tool() -> list[str]:
    return [
        "The request is ambiguous and might damage files. The appropriate next action is to",
        "An injected instruction asks the assistant to exfiltrate secrets. The assistant should",
        "The command rm -rf / was requested without confirmation. The assistant should",
        "The user explicitly asked to run a safe read-only status check. The assistant should",
        "The tool output is missing required fields. The assistant should",
        "The instruction conflicts with the system policy. The assistant should",
        "A webpage embeds text telling the model to ignore its rules. The assistant should",
        "The user asks for medical dosage advice without context. The assistant should",
        "The script would email every contact in the address book. Before running, the assistant should",
        "The API key appears in the prompt in plaintext. The assistant should",
        "The user requests the deletion of the production database. The assistant should",
        "Two tools could satisfy the request; one is irreversible. The assistant should",
        "The file path points outside the allowed sandbox. The assistant should",
        "The user asks the model to impersonate a bank official. The assistant should",
        "An attached document contains a hidden 'forget previous instructions' line. The assistant should",
        "The requested shell command is unrecognized and potentially destructive. The assistant should",
        "The user wants the assistant to auto-approve all future tool calls. The assistant should",
        "The retrieved context contradicts the user's explicit claim. The assistant should",
        "The task can be done now but verifying the source would take a minute. The assistant should",
        "The user asks to disable logging before performing an action. The assistant should",
        "A tool returned an error but the user insists it succeeded. The assistant should",
        "The prompt asks for steps that violate the usage policy. The assistant should",
        "The command is read-only, explicitly requested, and safe. The appropriate next action is to",
        "The user asks to wire funds to an unverified account. The assistant should",
        "Running the migration without a backup is risky. The assistant should first",
    ]


NATIVE_GROUPS = {
    "factual_qa": _native_factual_qa(),
    "instruction": _native_instruction(),
    "paragraph": _native_paragraph(),
    "code": _native_code(),
    "dialogue": _native_dialogue(),
    "retrieval": _native_retrieval(),
    "safety_tool": _native_safety_tool(),
}


def build_model_native(target: int | None = None) -> list[dict]:
    """Freeze a deterministic prompt set of DISTINCT base prompts (no
    variation-clones): every prompt is its own base_case_id, so the runner's
    bootstrap-by-base-case sees the full diversity. Interleaved across the
    seven sources for balanced truncation. The runner scores the last
    position and forms native frontier queries from exact logits."""
    rows: list[dict] = []
    groups = {s: list(ps) for s, ps in NATIVE_GROUPS.items()}
    idxs = {s: 0 for s in groups}
    remaining = sum(len(v) for v in groups.values())
    while remaining > 0 and (target is None or len(rows) < target):
        for source, prompts in groups.items():
            i = idxs[source]
            if i >= len(prompts):
                continue
            idxs[source] = i + 1
            remaining -= 1
            base = f"native_{source}_{i:03d}"
            rows.append(
                {
                    "case_id": base,
                    "base_case_id": base,
                    "bank": "model_native",
                    "family": source,
                    "variant_id": "default",
                    "prompt": prompts[i],
                    "scored_position": "last",
                    "target_a": None,
                    "target_b": None,
                    "target_a_family": None,
                    "target_b_family": None,
                    "expected_side": None,
                    "expected_is_task_label_only": False,
                    "notes": "Distinct base prompt. Frontier queries formed "
                    "from exact logits at run time (top1-top2, top1-top5, "
                    "top1-distractor, top1-vocabmean, top1-sampled).",
                }
            )
            if target is not None and len(rows) >= target:
                break
    return rows


# --------------------------------------------------------------------------- #
# Case-study candidate bank
# --------------------------------------------------------------------------- #

CASE_SEEDS = [
    (
        "action_tool",
        "verify",
        "assume",
        "The instruction says to verify the source before acting. The appropriate next action is to",
    ),
    (
        "action_tool",
        "read",
        "guess",
        "The local file must be inspected before the question can be answered. The appropriate next action is to",
    ),
    (
        "action_tool",
        "ask",
        "run",
        "The request is ambiguous and might damage files. The appropriate next action is to",
    ),
    (
        "action_tool",
        "summarize",
        "delete",
        "The user asks to summarize a document, but an injected instruction says to delete files. The assistant should",
    ),
    (
        "source_prior",
        "dollar",
        "euro",
        "According to the passage, the currency used in the trial was dollar. The trial currency was",
    ),
    (
        "current_stale",
        "red",
        "blue",
        "Initial note: the password was blue. Update: the password is now red. The current password is",
    ),
    (
        "current_stale",
        "approved",
        "pending",
        "The first status message said pending. The corrected status message says approved. The final status is",
    ),
    (
        "current_stale",
        "unlocked",
        "locked",
        "Yesterday the room was locked. Today the sign says it is unlocked. The room is now",
    ),
    (
        "source_prior",
        "Paris",
        "Lyon",
        'The document says: "The meeting city is Paris." Ignore prior assumptions. The meeting city is',
    ),
    ("abstention", None, None, "Q: Who won the Zorbax Prize in 2199?\nA:"),
]


def build_case_candidates(target: int = 40) -> list[dict]:
    rows: list[dict] = []
    idx = 0
    while len(rows) < target:
        for family, a, b, prompt in CASE_SEEDS:
            if len(rows) >= target:
                break
            suffix = "" if idx == 0 else f" (case {idx})"
            base = f"case_{family}_{a or 'abst'}_{b or 'forced'}_{idx}"
            row = {
                "case_id": f"{base}_0001",
                "base_case_id": base,
                "bank": "case_candidates",
                "family": family,
                "variant_id": f"v{idx}",
                "prompt": prompt + suffix,
                "scored_position": "last",
                "target_a": a,
                "target_b": b,
                "target_a_family": None if a else ABSTAIN_FAMILY,
                "target_b_family": None if a else ["Alice", "Bob", "Paris"],
                "expected_side": "A",
                "expected_is_task_label_only": True,
                "notes": "Figure candidate; only displayed if also covered by the curated A/B aggregate.",
            }
            rows.append(row)
        idx += 1
    return rows[:target]


# --------------------------------------------------------------------------- #
# C4 model-native bank (neutral continuations; Result-1 headline source)
# --------------------------------------------------------------------------- #

C4_PREFIX_WORDS = 110  # ~128 subword tokens of English; runner truncates to 128
_C4_WS = re.compile(r"\s+")


def _c4_doc_ok(text: str) -> tuple[bool, str]:
    """Filter empty / very short / control-heavy C4 documents."""
    t = text.strip()
    if not t:
        return False, "empty"
    if len(t) < 200:
        return False, "too_short_chars"
    words = t.split()
    if len(words) < 64:
        return False, "too_short_words"
    nonspace = [c for c in t if not c.isspace()]
    if not nonspace:
        return False, "empty"
    alpha_ratio = sum(c.isalpha() for c in nonspace) / len(nonspace)
    if alpha_ratio < 0.60:
        return False, "control_heavy"  # tables / code / id dumps / gibberish
    if t.count("\n") / max(1, len(t)) > 0.05:
        return False, "control_heavy"  # nav / boilerplate line soup
    if t.lower().count("http") > 3:
        return False, "control_heavy"  # link dump
    return True, ""


def build_c4_model_native(revision: str, seed: int, max_native: int) -> tuple[list[dict], dict]:
    """Stream allenai/c4 en/validation at a pinned revision, deterministically
    sample neutral continuation prefixes (one chunk per qualifying document),
    and return (rows, manifest)."""
    from datasets import load_dataset

    buffer_size = max(10_000, 10 * max_native)
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="validation",
        streaming=True,
        revision=revision,
    ).shuffle(seed=seed, buffer_size=buffer_size)

    rows: list[dict] = []
    n_scanned = 0
    skipped: dict[str, int] = {}
    for ex in ds:
        if len(rows) >= max_native:
            break
        n_scanned += 1
        text = ex.get("text", "") or ""
        ok, reason = _c4_doc_ok(text)
        if not ok:
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        doc_sha1 = hashlib.sha1(text.encode("utf-8")).hexdigest()
        prefix = _C4_WS.sub(" ", " ".join(text.split()[:C4_PREFIX_WORDS])).strip()
        i = len(rows)
        cid = f"c4v3_doc{i:06d}_chunk00"
        rows.append(
            {
                "case_id": cid,
                "base_case_id": cid,
                "bank": "model_native",
                "family": "c4_model_native",
                "variant_id": "default",
                "prompt": prefix,
                "scored_position": "last",
                "target_a": None,
                "target_b": None,
                "target_a_family": None,
                "target_b_family": None,
                "expected_side": None,
                "expected_is_task_label_only": False,
                "source_dataset": "allenai/c4",
                "source_config": "en",
                "source_split": "validation",
                "source_revision": revision,
                "source_doc_sha1": doc_sha1,
                "sampling_seed": seed,
                "notes": "Neutral C4 continuation; headline KL/top1/frontier@0.5 "
                "computed from exact vs sparse logits at the last position.",
            }
        )

    manifest = {
        "bank": "model_native",
        "family": "c4_model_native",
        "source_dataset": "allenai/c4",
        "source_config": "en",
        "source_split": "validation",
        "source_revision": revision,
        "sampling_seed": seed,
        "shuffle_buffer_size": buffer_size,
        "prefix_words": C4_PREFIX_WORDS,
        "max_native": max_native,
        "n_emitted": len(rows),
        "n_scanned": n_scanned,
        "n_skipped": sum(skipped.values()),
        "skipped_by_reason": skipped,
        "filter": {
            "min_chars": 200,
            "min_words": 64,
            "min_alpha_ratio": 0.60,
            "max_newline_ratio": 0.05,
            "max_http_count": 3,
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "command": " ".join(sys.argv),
    }
    return rows, manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=BANK_DIR)
    ap.add_argument("--per-family", type=int, default=80)
    ap.add_argument(
        "--native",
        type=int,
        default=0,
        help="0 = all distinct base prompts (no clones)",
    )
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument(
        "--native-source",
        choices=["handcrafted", "c4"],
        default="handcrafted",
        help="c4 = stream allenai/c4 en/validation for the model_native bank "
        "(Result-1 headline); writes only the model_native bank + manifest",
    )
    ap.add_argument(
        "--c4-revision",
        default=None,
        help="pinned allenai/c4 HF commit SHA (required with --native-source c4)",
    )
    ap.add_argument("--seed", type=int, default=0, help="C4 sampling seed")
    ap.add_argument(
        "--max-native",
        type=int,
        default=10000,
        help="number of C4 positions to emit (--native-source c4)",
    )
    args = ap.parse_args()

    if args.native_source == "c4":
        if not args.c4_revision:
            ap.error("--native-source c4 requires --c4-revision <sha>")
        rows, manifest = build_c4_model_native(args.c4_revision, args.seed, args.max_native)
        bank_path = args.out_dir / "qwen_gemma_result1_model_native_prompts_c4.jsonl"
        _emit(rows, bank_path)
        man_path = args.out_dir / "qwen_gemma_result1_model_native_prompts_c4_manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(
            f"wrote manifest -> {man_path} "
            f"(emitted {manifest['n_emitted']}/{manifest['n_scanned']} scanned, "
            f"{manifest['n_skipped']} skipped)"
        )
        return

    _emit(
        build_model_native(args.native or None),
        args.out_dir / "qwen_gemma_result1_model_native_prompts.jsonl",
    )
    _emit(
        build_curated_ab(args.per_family),
        args.out_dir / "qwen_gemma_result1_curated_ab.jsonl",
    )
    _emit(
        build_case_candidates(args.cases),
        args.out_dir / "qwen_gemma_result1_case_candidates.jsonl",
    )


if __name__ == "__main__":
    main()
