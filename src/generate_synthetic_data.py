"""Generate the deterministic PDF fixture set used by this project.

The PDFs intentionally vary their field labels, order, and whitespace. They
remain one page each and contain only synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from enum import IntEnum

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "raw_pdfs"



class LayoutVariant(IntEnum):
    STANDARD = 0 #The Investment Strategy: A balanced approach. It weighs financial returns equally alongside basic ESG metrics.
    FINANCIAL_FIRST = 1 #: Capital growth is the primary objective. The fund seeks maximum financial returns first, and treats ESG factors as secondary risk management tools.
    COMPLIANCE_FOCUS = 2 # Highly regulated. This strategy is typical for institutional funds (like pension funds) that must legally prove they meet strict sustainability mandates and operations statuses.
    FEE_PROMINENT = 3 # Low-cost targeting. This represents index funds or passive ETFs where the main selling point to the investor is how cheap the fund is to run compared to competitors.

@dataclass(frozen=True)
class FundFixture:
    fund_id: str
    filename: str
    fund_name: str
    esg: bool
    active: bool
    closed_quarter: str | None
    expense_ratio_percent: float
    aum_millions_usd: float
    layout_variant: LayoutVariant




FUNDS: tuple[FundFixture, ...] = (
    FundFixture("F001", "F001_evergreen_esg_equity.pdf", "Evergreen ESG Equity Fund", True, True, None, 0.45, 120, LayoutVariant.STANDARD),
    FundFixture("F002", "F002_horizon_sustainable_bond.pdf", "Horizon Sustainable Bond Fund", True, True, None, 0.38, 80, LayoutVariant.FINANCIAL_FIRST),
    FundFixture("F003", "F003_climate_transition_leaders.pdf", "Climate Transition Leaders Fund", True, True, None, 0.52, 60, LayoutVariant.COMPLIANCE_FOCUS),
    FundFixture("F004", "F004_green_infrastructure.pdf", "Green Infrastructure Fund", True, True, None, 0.65, 150, LayoutVariant.FEE_PROMINENT),
    FundFixture("F005", "F005_responsible_global_index.pdf", "Responsible Global Index Fund", True, True, None, 0.22, 200, LayoutVariant.STANDARD),
    FundFixture("F006", "F006_esg_income_opportunities.pdf", "ESG Income Opportunities Fund", True, True, None, 0.40, 90, LayoutVariant.FINANCIAL_FIRST),
    FundFixture("F007", "F007_sustainable_value.pdf", "Sustainable Value Fund", True, False, "Q3", 0.55, 75, LayoutVariant.COMPLIANCE_FOCUS),
    FundFixture("F008", "F008_esg_real_assets.pdf", "ESG Real Assets Fund", True, False, "Q3", 0.70, 110, LayoutVariant.FEE_PROMINENT),
    FundFixture("F009", "F009_esg_defensive_allocation.pdf", "ESG Defensive Allocation Fund", True, False, "Q4", 0.30, 65, LayoutVariant.STANDARD),
    FundFixture("F010", "F010_core_us_equity.pdf", "Core US Equity Fund", False, True, None, 0.12, 300, LayoutVariant.FINANCIAL_FIRST),
    FundFixture("F011", "F011_global_opportunities.pdf", "Global Opportunities Fund", False, True, None, 0.48, 85, LayoutVariant.COMPLIANCE_FOCUS),
    FundFixture("F012", "F012_income_balanced.pdf", "Income Balanced Fund", False, False, "Q3", 0.35, 95, LayoutVariant.FEE_PROMINENT),
)


def _status_text(fund: FundFixture) -> str:
    return "Active" if fund.active else f"Closed - {fund.closed_quarter}"


def _esg_text(fund: FundFixture) -> str:
    return "Yes" if fund.esg else "No"


def _field_lines(fund: FundFixture) -> list[str]:
    common = {
        "id": f"Fund ID: {fund.fund_id}",
        "esg": f"ESG: {_esg_text(fund)}",
        "type": f"Fund Type: {'ESG-Aligned' if fund.esg else 'Conventional'}",
        "status": f"Active Status: {_status_text(fund)}",
        "lifecycle": f"Fund Lifecycle: {_status_text(fund)}",
        "expense": f"Expense Ratio: {fund.expense_ratio_percent:.2f}%",
        "fee": f"Annual Fund Fee   {fund.expense_ratio_percent:.2f}%",
        "aum": f"AUM / Fund Size: ${fund.aum_millions_usd:.0f}M",
        "assets": f"Net Assets: ${fund.aum_millions_usd:.0f} million",
    }
    variants = (
        [common["id"], common["esg"], common["status"], common["expense"], common["aum"]],
        [common["aum"], common["id"], common["type"], common["fee"], common["lifecycle"]],
        [common["status"], common["assets"], common["id"], common["esg"], common["expense"]],
        [common["fee"], common["type"], common["aum"], common["lifecycle"], common["id"]],
    )
    try:
        return variants[fund.layout_variant]
    except IndexError as e:
        raise ValueError(f"Invalid layout_variant {fund.layout_variant} for fund {fund.fund_id}") from e


def _draw_wrapped_text(canvas: Canvas, text: str, x: float, y: float, max_width: float) -> float:
    """Draw a short line, wrapping only if a long fund name needs it."""
    words = text.split()
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if line and stringWidth(candidate, "Helvetica-Bold", 18) > max_width:
            canvas.drawString(x, y, line)
            y -= 24
            line = word
        else:
            line = candidate
    canvas.drawString(x, y, line)
    return y


def build_pdf(fund: FundFixture, destination: Path) -> None:
    """Write one legible, single-page PDF for *fund*."""
    canvas = Canvas(str(destination), pagesize=LETTER, pageCompression=1)
    page_width, page_height = LETTER
    left = 56
    top = page_height - 64

    canvas.setFillColor(HexColor("#14364A"))
    canvas.setFont("Helvetica-Bold", 10)
    canvas.drawString(left, top, "SYNTHETIC FUND FACT SHEET")
    canvas.setStrokeColor(HexColor("#3A8D9B"))
    canvas.setLineWidth(1.2)
    canvas.line(left, top - 10, page_width - left, top - 10)

    canvas.setFillColor(HexColor("#1B1B1B"))
    canvas.setFont("Helvetica-Bold", 18)
    title_y = _draw_wrapped_text(canvas, fund.fund_name, left, top - 42, page_width - (left * 2))

    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(HexColor("#5B6570"))
    canvas.drawString(left, title_y - 24, "Synthetic fixture - not an investment document")

    y = title_y - 72
    for index, line in enumerate(_field_lines(fund)):
        if index % 2 == 0:
            canvas.setFillColor(HexColor("#EEF5F6"))
            canvas.roundRect(left - 8, y - 14, page_width - (left * 2) + 16, 24, 3, fill=1, stroke=0)
        canvas.setFillColor(HexColor("#1B1B1B"))
        canvas.setFont("Helvetica", 12)
        canvas.drawString(left, y, line)
        try:
            y -= 38 if fund.layout_variant in (0, 2) else 42
        except ValueError as e:
            raise ValueError(f"Invalid layout_variant {fund.layout_variant} for fund {fund.fund_id}") from e

    canvas.setStrokeColor(HexColor("#D2D8DD"))
    canvas.line(left, 64, page_width - left, 64)
    canvas.setFillColor(HexColor("#5B6570"))
    canvas.setFont("Helvetica", 8)
    canvas.drawString(left, 48, f"Fixture reference: {fund.fund_id}")
    canvas.drawRightString(page_width - left, 48, "For pipeline testing only")
    canvas.save()


def generate_all(output_directory: Path = OUTPUT_DIRECTORY) -> list[Path]:
    """Generate all fixtures and return their paths in deterministic order."""
    output_directory.mkdir(parents=True, exist_ok=True)
    generated_paths: list[Path] = []
    for fund in FUNDS:
        destination = output_directory / fund.filename
        build_pdf(fund, destination)
        generated_paths.append(destination)
    return generated_paths


if __name__ == "__main__":
    paths = generate_all()
    print(f"Generated {len(paths)} synthetic fund PDFs in {OUTPUT_DIRECTORY}")
