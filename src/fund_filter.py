"""Deterministic filtering of funds from structured fields in extracted text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class FilterSpec:
    is_esg: bool | None = None
    status: str | None = None


@dataclass(frozen=True)
class FundMetadata:
    name: str
    is_esg: bool
    status: str
    closed_quarter: str | None

#when we do not use ?,it means Not only do I want to verify this text matches my rules, but I also want you to extract it, save it in RAM, and hand it back to me as a variable so I can use it in my code.
_ESG_PATTERN = re.compile(r"^ESG:\s*(Yes|No)\s*$", re.IGNORECASE) #^ forces the text start at the absolute beginning of the line, and $ forces the text to end at the absolute end of the line. This ensures that we are matching the entire line and not just a substring.
_FUND_TYPE_PATTERN = re.compile(
    r"^Fund Type:\s*(ESG-Aligned|Conventional)\s*$", re.IGNORECASE # it is either gonna be ESG-Aligned|Conventional 
)
_STATUS_PATTERN = re.compile(
    r"^(?:Active Status|Fund Lifecycle):\s*(Active|Closed\s*-\s*Q[1-4])\s*$", #means Non-Capturing Group. The computer checks if the line starts with either label, but it tells Python: "I don't care which label was used, don't save the label text."
    re.IGNORECASE,
)
# ? at the very beginning of the pattern indicates a non-capturing group. It checks if the text says either "Active Status" or "Fund Lifecycle", but it will not allocate memory to return that specific label text. This keeps your matching fast and clean.

def parse_fund_metadata(text: str) -> FundMetadata:
    """Parse required structured metadata from a fund PDF's extracted text."""
    lines = [line.strip() for line in text.splitlines() if line.strip()] #This function completely ignores normal spaces and tabs. It searches only for the boundaries where a row ends and a new row begins. (['Fund ID:  F001', 'ESG: Yes']) while with tetx.split it would be ['Fund', 'ID:', 'F001', 'ESG:', 'Yes']
    if len(lines) < 2:
        raise ValueError("Fund text does not contain a fund name")

    name = lines[1]
    is_esg: bool | None = None
    status: str | None = None
    closed_quarter: str | None = None

    for line in lines:
        esg_match = _ESG_PATTERN.match(line)
        type_match = _FUND_TYPE_PATTERN.match(line)
        status_match = _STATUS_PATTERN.match(line)

        if esg_match:
            is_esg = esg_match.group(1).lower() == "yes" #if the is_esg is Yes, then it will be True, otherwise it will be False.
             #Group(0): Always represents the entire line of text that matched the pattern.
        elif type_match:
            is_esg = type_match.group(1).lower() == "esg-aligned" #iif the is_esg is ESG-Aligned, then it will be True, otherwise it will be False.
             #Group(1): Represents only the specific text caught inside the first pair of parentheses—in this case, either the word "Yes" or the word "No".
        elif status_match:
            raw_status = status_match.group(1).lower()
            status = "active" if raw_status == "active" else "closed"
            quarter_match = re.search(r"Q[1-4]", raw_status, re.IGNORECASE)
            closed_quarter = quarter_match.group(0).upper() if quarter_match else None

    if is_esg is None:
        raise ValueError(f"Missing ESG field for fund: {name}")
    if status is None:
        raise ValueError(f"Missing active-status field for fund: {name}")

    return FundMetadata(
        name=name,
        is_esg=is_esg,
        status=status,
        closed_quarter=closed_quarter,
    )


def filter_funds(
    extracted_funds: Iterable[Mapping[str, object]], filter_spec: FilterSpec
) -> list[str]:
    """Return qualifying fund names in source order using exact parsed fields."""
    qualifying_funds: list[str] = []

    for extracted_fund in extracted_funds:
        text = extracted_fund.get("text")
        if not isinstance(text, str):
            raise ValueError("Each extracted fund must contain a string 'text' field") 

        fund = parse_fund_metadata(text)
        if filter_spec.is_esg is not None and fund.is_esg != filter_spec.is_esg: # "All-or-Nothing" selection policy
            continue
        if filter_spec.status is not None and fund.status != filter_spec.status.lower():
            continue
        qualifying_funds.append(fund.name)

    return qualifying_funds
# If they do not match (e.g., the user wants ESG, but the fund is Conventional), the code triggers continue. Python drops the current fund right there, skips the rest of the checks, and starts parsing the next document.

# we use iterable instead of list because we want to be able to handle any iterable input, not just lists. This allows for more flexibility in how the function can be used, such as with generators or other iterable types.
#we use Mapping instead of dict because we want to allow for any mapping type, not just dictionaries. This allows for more flexibility in how the function can be used, such as with custom mapping types or other dictionary-like objects.

