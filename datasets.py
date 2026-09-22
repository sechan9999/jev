"""Small labeled datasets for the benchmark.

Each case carries:
  * ``query``   -- the text the model reads
  * ``answers`` -- dict of {answer -> meaning shown to the model}
  * ``label``   -- the expected answer (used to score correctness)

The answer sets always include an escape hatch ("other") so a case that
fits none of the named categories is not forced into a wrong one -- the
article calls this out as necessary because restricted softmax otherwise
puts all probability mass on the listed choices.
"""

from __future__ import annotations

import itertools

# --------------------------------------------------------------------------
# 1) Support ticket routing
# --------------------------------------------------------------------------
SUPPORT_ANSWERS = {
    "billing": "billing questions, charges, refunds, and payment problems",
    "technical": "product errors, bugs, outages, and technical failures",
    "account": "login, password, and account access problems",
    "other": "anything that does not fit the categories above",
}

SUPPORT_CASES = [
    ("I was charged twice for the same subscription.", "billing"),
    ("The app crashes every time I upload a file.", "technical"),
    ("I can't log in even after resetting my password.", "account"),
    ("Please refund the duplicate payment on my card.", "billing"),
    ("Getting a 500 error when I open the dashboard.", "technical"),
    ("My two-factor codes never arrive by SMS.", "account"),
    ("Why was I billed after cancelling last month?", "billing"),
    ("The export button does nothing and the page freezes.", "technical"),
    ("I need to update the email address on my account.", "account"),
    ("Do you sell gift cards for the holidays?", "other"),
]

# --------------------------------------------------------------------------
# 2) Candidate screening (resume vs role)
# --------------------------------------------------------------------------
SCREENING_ANSWERS = {
    "advance": "clearly meets the core requirements; move to interview",
    "review": "borderline; a human recruiter should review",
    "reject": "clearly does not meet the core requirements",
    "other": "not enough information to decide",
}

SCREENING_CASES = [
    ("Role: Senior Python engineer, 5+ yrs. Candidate: 8 yrs Python, led backend teams.", "advance"),
    ("Role: Senior Python engineer, 5+ yrs. Candidate: 1 yr Python, bootcamp grad.", "reject"),
    ("Role: Data scientist, ML in production. Candidate: strong ML research, no prod experience.", "review"),
    ("Role: Frontend dev, React. Candidate: 6 yrs React, shipped 3 large apps.", "advance"),
    ("Role: DevOps, Kubernetes. Candidate: only Windows desktop support background.", "reject"),
    ("Role: Product analyst, SQL. Candidate: strong SQL, some dashboarding, unclear domain fit.", "review"),
    ("Role: Security engineer. Candidate: resume is blank except a name.", "other"),
    ("Role: Mobile iOS dev. Candidate: 4 yrs Swift, App Store apps with 1M+ users.", "advance"),
    ("Role: Staff data engineer. Candidate: 10 yrs but only ETL in legacy tools.", "review"),
    ("Role: ML platform lead. Candidate: marketing manager, no engineering.", "reject"),
]

# --------------------------------------------------------------------------
# 3) Expense review (approve / flag / deny)
# --------------------------------------------------------------------------
EXPENSE_ANSWERS = {
    "approve": "within policy; approve automatically",
    "flag": "possibly out of policy; route to a manager",
    "deny": "clearly violates policy; deny",
    "other": "cannot classify from the description",
}

EXPENSE_CASES = [
    ("$12 taxi from airport to hotel with receipt attached.", "approve"),
    ("$4,500 first-class flight, no manager pre-approval.", "flag"),
    ("$300 personal spa treatment charged to company card.", "deny"),
    ("$45 team lunch, 4 attendees, receipt attached.", "approve"),
    ("$1,200 conference ticket, receipt attached, pre-approved.", "approve"),
    ("$800 hotel, one night, city rate cap is $250.", "flag"),
    ("$60 alcohol-only bar tab, no business attendees listed.", "deny"),
    ("$9 parking fee with receipt.", "approve"),
    ("$2,000 'miscellaneous', no receipt, no description.", "flag"),
    ("$500 gift card purchased for personal use.", "deny"),
]


def _expand(answers: dict[str, str], cases: list[tuple[str, str]], domain: str) -> list[dict]:
    return [
        {"domain": domain, "query": query, "answers": answers, "label": label}
        for query, label in cases
    ]


def all_cases(n: int = 100) -> list[dict]:
    """Return ``n`` cases by cycling through the three domains.

    The base sets are small and hand-labeled; cycling gives a repeatable
    benchmark of any size without inventing unlabeled data.
    """
    base = (
        _expand(SUPPORT_ANSWERS, SUPPORT_CASES, "support")
        + _expand(SCREENING_ANSWERS, SCREENING_CASES, "screening")
        + _expand(EXPENSE_ANSWERS, EXPENSE_CASES, "expense")
    )
    return list(itertools.islice(itertools.cycle(base), n))
