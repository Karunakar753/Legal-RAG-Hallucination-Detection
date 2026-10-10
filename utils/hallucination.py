"""Heuristic detection of *potentially* unsupported claims in an LLM answer.

Each sentence-level claim is scored against the retrieved evidence with:
  * semantic similarity (embedding cosine vs. evidence passages/sentences),
  * lexical coverage (share of the claim's content words found in the evidence),
  * hard checks: case names, legal provisions and numbers/dates must literally appear in the evidence,
  * citation checks: cited pages must exist in the retrieved evidence.
This flags claims for human review. It cannot prove an answer correct or incorrect.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np


EmbedFn = Callable[[list[str]], np.ndarray]  # returns L2-normalised rows

SUPPORT_THRESHOLD = 0.50  # claim score below this => flagged
SEM_LOW, SEM_HIGH = 0.15, 0.60  # cosine range mapped to 0..1 (heuristic, tune per model)
MIN_CLAIM_WORDS = 4

_CITE_RE = re.compile(r"[\[(]\s*(?:Source\s*\d+\s*[|,]?\s*)?Pages?\s*(\d+(?:\s*[-\u2013,]\s*\d+)*)\s*[\])]", re.I)
_ABBREV = r"".join(
    rf"(?<!\b{a}\.)" for a in ("v", "vs", "No", "Sec", "Art", "Dr", "Mr", "Mrs", "Ms", "Smt", "Co", "Ltd", "Inc", "Hon", "J")
) + r"(?<!\b[A-Z]\.)"
_SENT_RE = re.compile(_ABBREV + r"(?<=[.!?])\s+(?=[A-Z\[(\"'])")
_BULLET_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")
_NON_CLAIM_RE = re.compile(
    r"\b(insufficient|does not (?:contain|provide|mention|state|specify|address|cover)|"
    r"not (?:enough|sufficient) (?:information|evidence)|cannot be determined|no (?:information|evidence))\b",
    re.I,
)
_TOKEN = r"(?:[A-Z]\.|[A-Z][\w&'\u2019-]*)"
_PARTY = rf"{_TOKEN}(?:\s+(?:(?:of|the|&)\s+)?{_TOKEN})*"
_CASE_RE = re.compile(rf"({_PARTY})\s+(?:v|vs|versus)\b\.?\s+({_PARTY})")
_LEAD_RE = re.compile(r"^(?:(?:in|the|see|as|under|per|and|that|also|however|thus|but)\s+)+", re.I)
_PROVISION_RE = re.compile(r"\b(?:section|article|rule|clause|order)\s+\d+[a-z]?(?:\([a-z0-9]+\))*")
_NUMBER_RE = re.compile(r"\d[\d./-]*\d|\d")
_STOP = frozenset(
    "that this with from have been were which their there would could should about into also than then them "
    "these those such other under upon over after before while where when what whom whose shall will your "
    "court held said case any all are for has had its not the and was but may can only".split()
)


@dataclass
class ClaimCheck:
    claim: str
    cited_pages: list[int]
    semantic_score: float  # raw cosine to best evidence unit
    lexical_score: float
    score: float  # combined 0..1
    best_page: Optional[int]
    reasons: list[str] = field(default_factory=list)
    supported: bool = True


@dataclass
class HallucinationReport:
    overall_score: Optional[float]  # None => nothing to score (e.g. answer declined)
    verdict: str
    claims: list[ClaimCheck]
    cited_pages: list[int]
    invalid_citations: list[int]
    note: str = ""

    @property
    def unsupported_claims(self) -> list[ClaimCheck]:
        return [c for c in self.claims if not c.supported]


def _norm(text: str) -> str:
    t = text.lower()
    t = re.sub(r"(?<=\d),(?=\d)", "", t)  # 1,00,000 -> 100000
    t = re.sub(r"\b(?:versus|vs?)\b\.?", "v", t)
    t = re.sub(r"\bsec(?:tion)?s?\b\.?", "section", t)
    t = re.sub(r"\b(?:articles?|arts?)\b\.?", "article", t)
    t = re.sub(r"\b(rule|clause)s\b", r"\1", t)
    t = re.sub(r"\s+\(", "(", t)
    return re.sub(r"\s+", " ", t).strip()


def _pages_in(text: str) -> list[int]:
    return [int(n) for group in _CITE_RE.findall(text) for n in re.findall(r"\d+", group)]


def _strip_citations(text: str) -> str:
    return re.sub(r"\s+", " ", _CITE_RE.sub(" ", text)).strip()


def split_claims(answer: str) -> list[tuple[str, str]]:
    """Return (claim_with_citations, claim_plain) pairs; citation-only fragments join the previous claim."""
    fragments: list[str] = []
    for line in answer.splitlines():
        line = _BULLET_RE.sub("", line).strip()
        if line:
            fragments.extend(f.strip() for f in _SENT_RE.split(line) if f.strip())
    merged: list[str] = []
    for frag in fragments:
        m = _CITE_RE.match(frag)
        while m and merged:  # a citation that trails the previous sentence's full stop
            merged[-1] += " " + m.group(0)
            frag = frag[m.end():].lstrip(" .;:")
            m = _CITE_RE.match(frag)
        if not frag:
            continue
        if merged and not _strip_citations(frag).strip(" .;:"):
            merged[-1] += " " + frag
        else:
            merged.append(frag)
    out = []
    for frag in merged:
        plain = _strip_citations(frag)
        if len(plain.split()) >= MIN_CLAIM_WORDS and not _NON_CLAIM_RE.search(plain):
            out.append((frag, plain))
    return out


def _stems(text: str) -> set[str]:
    return {w[:6] for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in _STOP}


def find_missing_specifics(claim_plain: str, evidence_norm: str) -> list[str]:
    """Case names, provisions and numbers/dates in the claim that never appear in the evidence."""
    missing: list[str] = []
    masked = claim_plain
    for match in _CASE_RE.finditer(claim_plain):
        name = _LEAD_RE.sub("", match.group(0)).strip(" .,;:")
        if _norm(name) not in evidence_norm:
            missing.append(f"case name '{name}'")
    masked = _CASE_RE.sub(" ", masked)
    norm_claim = _norm(masked)
    for prov in dict.fromkeys(_PROVISION_RE.findall(norm_claim)):
        if prov not in evidence_norm:
            missing.append(f"provision '{prov}'")
    norm_claim = _PROVISION_RE.sub(" ", norm_claim)
    seen: set[str] = set()
    for token in _NUMBER_RE.findall(norm_claim):
        for part in re.split(r"[./-]", token):
            if part and part not in seen:
                seen.add(part)
                if not re.search(rf"(?<!\d){re.escape(part)}(?!\d)", evidence_norm):
                    missing.append(f"number/date '{part}'")
    return missing


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if len(s.split()) >= 3]


def check_answer(
    answer: str,
    evidence: Sequence,
    embed_fn: EmbedFn,
    support_threshold: float = SUPPORT_THRESHOLD,
) -> HallucinationReport:
    """`evidence` items need `.page` and `.text` (Evidence or Chunk)."""
    answer = (answer or "").strip()
    evidence_pages = {int(e.page) for e in evidence}
    cited = sorted(set(_pages_in(answer)))
    invalid = [p for p in cited if p not in evidence_pages]

    claims = split_claims(answer) if answer else []
    if not claims or not evidence:
        return HallucinationReport(None, "No factual claims to verify", [], cited, invalid,
                                   "The answer declined or contained no checkable claims.")

    units: list[tuple[str, int]] = []
    for e in evidence:
        units.append((e.text, int(e.page)))
        units.extend((s, int(e.page)) for s in _sentences(e.text))
    unit_emb = np.asarray(embed_fn([u[0] for u in units]), dtype=np.float32)
    claim_emb = np.asarray(embed_fn([c[1] for c in claims]), dtype=np.float32)
    sims = claim_emb @ unit_emb.T

    evidence_norm = _norm(" ".join(e.text for e in evidence))
    evidence_stems = _stems(" ".join(e.text for e in evidence))

    checks: list[ClaimCheck] = []
    for i, (full, plain) in enumerate(claims):
        best = int(np.argmax(sims[i]))
        cosine = float(sims[i][best])
        sem = float(np.clip((cosine - SEM_LOW) / (SEM_HIGH - SEM_LOW), 0.0, 1.0))
        stems = _stems(plain)
        lex = len(stems & evidence_stems) / len(stems) if stems else sem
        score = 0.6 * sem + 0.4 * lex

        reasons = [f"Not found in evidence: {m}" for m in find_missing_specifics(plain, evidence_norm)]
        pages = sorted(set(_pages_in(full)))
        bad_pages = [p for p in pages if p not in evidence_pages]
        if bad_pages:
            reasons.append("Cites page(s) " + ", ".join(map(str, bad_pages)) + " that are not in the retrieved evidence")
        score *= 0.5 ** min(len(reasons), 3)
        if score < support_threshold and not reasons:
            reasons.append("Weak similarity to any retrieved passage")
        checks.append(ClaimCheck(full, pages, cosine, lex, float(score), units[best][1], reasons,
                                 supported=score >= support_threshold and not reasons))

    overall = float(np.mean([c.score for c in checks]))
    flagged = sum(not c.supported for c in checks)
    if overall >= 0.75 and flagged == 0:
        verdict = "High support"
    elif overall >= 0.50:
        verdict = "Moderate support - review flagged claims"
    else:
        verdict = "Low support - answer may contain unsupported claims"
    return HallucinationReport(overall, verdict, checks, cited, invalid,
                               "Heuristic check; flagged items are potential issues, not proof of error.")
