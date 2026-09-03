import re


def _is_text_only(doc):
    """Return True if the question has no image content (text-only)."""
    img = doc.get("image")
    return not img or len(str(img)) <= 100


def process_docs_exact_match(dataset):
    """Filter to text-only exactMatch questions."""
    return dataset.filter(
        lambda x: x["answer_type"] == "exactMatch" and _is_text_only(x)
    )


def process_docs_multiple_choice(dataset):
    """Filter to text-only multipleChoice questions."""
    return dataset.filter(
        lambda x: x["answer_type"] == "multipleChoice" and _is_text_only(x)
    )


def doc_to_text(doc):
    """Format the question with an answer prompt.

    For exactMatch: ask the model to put the final answer after ####.
    For multipleChoice: the choices are already in the question text;
    ask for the letter answer.
    """
    question = doc["question"]
    if doc["answer_type"] == "exactMatch":
        return (
            f"{question}\n\n"
            f"Please solve this problem and put your final answer after \"#### \"."
        )
    else:
        return (
            f"{question}\n\n"
            f"The answer is the letter of the correct choice. "
            f"Let's think step by step, then state \"The answer is (X)\"."
        )
