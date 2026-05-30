from app.grounding import grounding_overlap, is_grounded
from app.models import Hit


def _hits(text: str) -> list[Hit]:
    return [Hit(chunk_id=1, document_id=1, text=text, score=1.0)]


def test_fully_grounded_answer_scores_high():
    context = "Refunds are processed within five business days to the original payment method."
    answer = "Refunds are processed within five business days."

    assert grounding_overlap(answer, _hits(context)) > 0.8
    assert is_grounded(answer, _hits(context)) is True


def test_hallucinated_answer_scores_low():
    context = "Refunds are processed within five business days."
    answer = "Our CEO is named Alexander and the company was founded in Berlin in 1998."

    assert grounding_overlap(answer, _hits(context)) < 0.2
    assert is_grounded(answer, _hits(context)) is False


def test_no_hits_is_never_grounded():
    assert is_grounded("anything", []) is False
