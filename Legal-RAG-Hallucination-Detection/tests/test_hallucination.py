from dataclasses import dataclass

from utils.embeddings import embed_texts
from utils.hallucination import check_answer, find_missing_specifics, split_claims, _norm
from utils.prompts import INSUFFICIENT_EVIDENCE_MESSAGE


@dataclass
class Ev:
    page: int
    text: str


EVIDENCE = [
    Ev(1, "State of Kerala v. Ramesh Kumar. The High Court convicted the accused under Section 302 of the "
          "Indian Penal Code and the trial court sentenced him to life imprisonment on 12 March 2018."),
    Ev(3, "For these reasons the appeal is dismissed and the conviction under Section 302 is upheld. "
          "The sentence of life imprisonment is confirmed. No order as to costs is made."),
]


def _check(fake_model, answer):
    return check_answer(answer, EVIDENCE, lambda texts: embed_texts(fake_model, texts))


def test_supported_answer_scores_high(fake_model):
    report = _check(fake_model, "The appeal is dismissed and the conviction under Section 302 is upheld [Page 3]. "
                                "The accused was sentenced to life imprisonment on 12 March 2018 [Page 1].")
    assert report.overall_score >= 0.6 and not report.unsupported_claims and not report.invalid_citations


def test_invented_facts_are_flagged(fake_model):
    report = _check(fake_model, "The appeal is dismissed and the conviction is upheld [Page 3]. "
                                "The Court relied on Sharma v. Union of India and Section 498A, awarding Rs 50000 [Page 7].")
    assert len(report.unsupported_claims) == 1
    reasons = " ".join(report.unsupported_claims[0].reasons)
    assert "Sharma v. Union of India" in reasons and "498a" in reasons and "50000" in reasons
    assert report.invalid_citations == [7]
    assert report.overall_score < 0.9


def test_declined_answer_has_no_claims(fake_model):
    report = _check(fake_model, INSUFFICIENT_EVIDENCE_MESSAGE)
    assert report.overall_score is None and report.claims == []


def test_specifics_detection():
    ev = _norm(" ".join(e.text for e in EVIDENCE))
    assert find_missing_specifics("The accused was convicted under Sec. 302 in State of Kerala vs. Ramesh Kumar.", ev) == []
    assert find_missing_specifics("Held in 2021 under Section 304 of the Act.", ev)
    assert find_missing_specifics("In Ramesh v. State of Goa the court agreed.", ev)


def test_claim_splitting_keeps_case_names_and_citations_together():
    claims = split_claims("The Court followed State v. Kumar. [Page 2] It dismissed the appeal on 5 August 2020.\n"
                          "- Costs were not awarded [Page 3].")
    assert len(claims) == 3
    assert "State v. Kumar" in claims[0][0] and "[Page 2]" in claims[0][0] and "[Page" not in claims[0][1]
