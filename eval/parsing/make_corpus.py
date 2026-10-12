"""Generate a small corpus of PDFs whose true contents we know, to measure table extraction.

Run (isolated environment, nothing is added to the project):
    uv run --with reportlab --with matplotlib --with pymupdf python eval/parsing/make_corpus.py

Writes eval/parsing/corpus/*.pdf plus truth.json (the exact table cells and chart labels).
The hard cases are the ones real filings contain: borderless financial tables, merged headers,
row spans, tables that continue over several pages, wide tables, charts, and scanned pages.
"""

import io
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pymupdf  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import mm  # noqa: E402
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)  # noqa: E402

OUT = Path(__file__).parent / "corpus"
STYLES = getSampleStyleSheet()
random.seed(7)  # the corpus must be identical on every run

truth: dict[str, dict] = {}


def money(n: int) -> str:
    return f"({abs(n):,})" if n < 0 else f"{n:,}"


def build(name: str, flowables: list, page=A4, tables=None, labels=None, note=""):
    path = OUT / f"{name}.pdf"
    SimpleDocTemplate(
        str(path),
        pagesize=page,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    ).build(flowables)
    truth[name] = {"tables": tables or [], "labels": labels or [], "note": note}


def para(text: str):
    return Paragraph(text, STYLES["BodyText"])


# --- 1. a plain bordered grid -----------------------------------------------------------
def grid_simple():
    rows = [["Region", "Customers", "Contracts", "Renewal rate", "Notes"]]
    for region in [
        "North America",
        "Europe",
        "Asia Pacific",
        "Latin America",
        "Middle East",
        "Africa",
        "Nordics",
        "Benelux",
    ]:
        rows.append(
            [
                region,
                f"{random.randint(100, 9000):,}",
                f"{random.randint(10, 900):,}",
                f"{random.randint(80, 99)}.{random.randint(0, 9)}%",
                random.choice(["stable", "growing", "at risk"]),
            ]
        )
    t = Table(rows, colWidths=[40 * mm, 30 * mm, 30 * mm, 30 * mm, 30 * mm])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
            ]
        )
    )
    build(
        "grid_simple",
        [para("Customer base by region."), Spacer(1, 6), t],
        tables=[rows],
    )


# --- 2. borderless financial statement (what a 10-K really looks like) --------------------
def borderless_financial():
    rows = [["", "2024", "2023", "2022"]]
    body = [
        ("Revenues", 307394, 282836, 257637),
        ("Cost of revenues", -133332, -126203, -110939),
        ("Gross profit", 174062, 156633, 146698),
        ("Research and development", -45427, -39500, -39500),
        ("Sales and marketing", -27917, -27917, -26567),
        ("General and administrative", -14188, -15724, -15724),
        ("Total operating expenses", -87532, -83141, -81791),
        ("Operating income", 86530, 73492, 64907),
        ("Other income, net", 7425, 1424, -3514),
        ("Income before income taxes", 93955, 74916, 61393),
        ("Provision for income taxes", -19697, -11922, -11356),
        ("Net income", 74258, 62994, 50037),
    ]
    for label, *vals in body:
        rows.append([label, *[money(v) for v in vals]])
    t = Table(rows, colWidths=[75 * mm, 28 * mm, 28 * mm, 28 * mm])
    t.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.black),
                ("LINEABOVE", (0, 3), (-1, 3), 0.4, colors.black),
                ("LINEABOVE", (0, 12), (-1, 12), 0.4, colors.black),
                ("FONT", (0, 3), (-1, 3), "Helvetica-Bold", 9),
                ("FONT", (0, 12), (-1, 12), "Helvetica-Bold", 9),
            ]
        )
    )
    build(
        "borderless_financial",
        [
            para("<b>CONSOLIDATED STATEMENTS OF INCOME</b>"),
            para("(in millions, except per share amounts)"),
            Spacer(1, 6),
            t,
        ],
        tables=[rows],
    )


# --- 3. merged header spanning the year columns ------------------------------------------
def merged_header():
    rows = [["Segment", "Fiscal year", "", ""], ["", "2024", "2023", "2022"]]
    for seg in ["Search", "Cloud", "Subscriptions", "Other bets"]:
        rows.append([seg, *[f"{random.randint(1000, 99000):,}" for _ in range(3)]])
    t = Table(rows, colWidths=[45 * mm, 30 * mm, 30 * mm, 30 * mm])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("SPAN", (1, 0), (3, 0)),
                ("SPAN", (0, 0), (0, 1)),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
            ]
        )
    )
    # truth lists each distinct cell text once: merged cells appear a single time
    truth_rows = [["Segment", "Fiscal year"], ["2024", "2023", "2022"], *rows[2:]]
    build(
        "merged_header",
        [para("Revenue by segment (USD millions)."), Spacer(1, 6), t],
        tables=[truth_rows],
    )


# --- 4. a table that continues over several pages, header repeated ------------------------
def multipage():
    rows = [["Account", "Q1", "Q2", "Q3", "Q4"]]
    for i in range(70):
        rows.append(
            [
                f"Account {i + 1:03d}",
                *[f"{random.randint(100, 99999):,}" for _ in range(4)],
            ]
        )
    t = Table(
        rows, colWidths=[40 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm], repeatRows=1
    )
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
            ]
        )
    )
    build(
        "multipage",
        [para("Quarterly detail by account."), Spacer(1, 6), t],
        tables=[rows],
        note="header row repeats on every page; truth has it once",
    )


# --- 5. wide table, small font ------------------------------------------------------------
def wide():
    head = [
        "Company",
        "Rev",
        "Growth",
        "EBITDA",
        "Margin",
        "Churn",
        "googogogaaga",
        "gibberish",
        "acquisitionDebt",
        "Cash",
        "Net lev.",
        "Buyback",
        "Div.",
    ]
    rows = [head]
    for name in [
        "Visa",
        "Mastercard",
        "PayPal",
        "Adyen",
        "Block",
        "Fiserv",
        "Global Pay",
        "Toast",
    ]:
        rows.append(
            [
                name,
                f"{random.randint(1, 40)}.{random.randint(0, 9)}",
                f"{random.randint(1, 30)}%",
                f"{random.randint(1, 20)}.{random.randint(0, 9)}",
                f"{random.randint(10, 60)}%",
                f"{random.randint(1, 30)}.{random.randint(0, 9)}",
                f"{random.randint(1, 30)}.{random.randint(0, 9)}",
                f"{random.randint(0, 4)}.{random.randint(0, 9)}x",
                f"{random.randint(0, 9)}.{random.randint(0, 9)}",
                f"{random.randint(0, 3)}.{random.randint(0, 9)}",
            ]
        )
    t = Table(rows, colWidths=[27 * mm] + [17 * mm] * 9)
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.4, colors.black),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 6.5),
            ]
        )
    )
    build(
        "wide",
        [para("Peer comparison."), Spacer(1, 6), t],
        page=landscape(A4),
        tables=[rows],
    )


# --- 6. row spans (a category cell covering several rows) ---------------------------------
def rowspan():
    rows = [["Category", "Item", "Amount"]]
    spec = [
        ("Assets", ["Cash", "Receivables", "Inventory"]),
        ("Liabilities", ["Payables", "Debt"]),
        ("Equity", ["Capital", "Retained earnings", "Reserves"]),
    ]
    spans, r = [], 1
    for cat, items in spec:
        for i, item in enumerate(items):
            rows.append([cat if i == 0 else "", item, f"{random.randint(100, 9999):,}"])
        if len(items) > 1:
            spans.append(("SPAN", (0, r), (0, r + len(items) - 1)))
        r += len(items)
    t = Table(rows, colWidths=[35 * mm, 50 * mm, 30 * mm])
    t.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                *spans,
            ]
        )
    )
    build("rowspan", [para("Balance sheet extract."), Spacer(1, 6), t], tables=[rows])


# --- 7. text only (to count false table detections) ---------------------------------------
def text_only():
    body = [
        para(
            "Alphabet reported higher revenue this year, driven by advertising and cloud services. "
            "Management noted continued investment in artificial intelligence infrastructure, and the "
            "board authorised a new share repurchase programme. Regulatory proceedings in the United "
            "States and Europe remain ongoing and could affect future results. " * 2
        )
        for _ in range(6)
    ]
    build("text_only", body)


# --- 8/9. charts: vector (text layer holds the labels) and raster (pixels only) -----------
def _chart():
    fig, ax = plt.subplots(figsize=(6, 3.4))
    years = [2019, 2020, 2021, 2022, 2023, 2024]
    ax.plot(years, [161, 182, 258, 283, 307, 350], marker="o", label="Total revenue")
    ax.plot(years, [100, 104, 149, 162, 175, 198], marker="s", label="Search")
    ax.set_title("Revenue by year")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel("USD billions")
    ax.legend()
    fig.canvas.draw()
    labels = [
        "Revenue by year",
        "Fiscal year",
        "USD billions",
        "Total revenue",
        "Search",
        *[str(y) for y in years],
        *[t.get_text() for t in ax.get_yticklabels() if t.get_text()],
    ]
    return fig, labels


def charts():
    plt.rcParams["pdf.fonttype"] = (
        42  # real TrueType text, so the text layer is extractable
    )
    fig, labels = _chart()
    fig.text(0.5, 0.01, "Figure 3: Revenue growth, 2019 to 2024.", ha="center")
    fig.savefig(OUT / "chart_vector.pdf", format="pdf")
    truth["chart_vector"] = {
        "tables": [],
        "labels": [*labels, "Figure 3: Revenue growth, 2019 to 2024."],
        "note": "vector chart: every label is real text",
    }
    png = io.BytesIO()
    fig.savefig(png, format="png", dpi=150)
    plt.close(fig)
    png.seek(0)
    caption = "Figure 3: Revenue growth, 2019 to 2024."
    build(
        "chart_raster",
        [
            para(caption),
            Image(png, width=150 * mm, height=85 * mm),
            para("Revenue grew every year over the period shown."),
        ],
        labels=labels,
        note="raster chart: only the caption is real text; the chart labels are pixels",
    )
    truth["chart_raster"]["caption"] = caption


# --- 10. scanned versions: pages rendered to images, NO text layer ------------------------
def scanned(source: str):
    src = pymupdf.open(OUT / f"{source}.pdf")
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=150)
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, pixmap=pix)
    out.save(OUT / f"scanned_{source}.pdf")
    truth[f"scanned_{source}"] = {
        **truth[source],
        "note": "image-only copy of " + source,
    }


# Two grades of "photocopier" damage. mild: a decent office scan. harsh: a poor fax or phone photo.
DAMAGE = {
    "mild": {"dpi": 150, "rotate": 0.6, "noise": 6, "blur": 0.0, "jpeg": 60},
    "harsh": {"dpi": 100, "rotate": 1.5, "noise": 14, "blur": 0.8, "jpeg": 30},
}


def degraded(source: str, grade: str):
    """Image-only copy with realistic scan damage: skew, sensor noise, blur, JPEG artefacts."""
    import numpy as np
    from PIL import Image, ImageFilter

    spec = DAMAGE[grade]
    rng = np.random.RandomState(11)
    src = pymupdf.open(OUT / f"{source}.pdf")
    out = pymupdf.open()
    for page in src:
        pix = page.get_pixmap(dpi=spec["dpi"])
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = img.rotate(
            spec["rotate"], resample=Image.BICUBIC, fillcolor=(255, 255, 255)
        )
        if spec["blur"]:
            img = img.filter(ImageFilter.GaussianBlur(spec["blur"]))
        arr = np.asarray(img).astype("float32")
        arr += rng.normal(0, spec["noise"], arr.shape)
        img = Image.fromarray(np.clip(arr, 0, 255).astype("uint8"))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=spec["jpeg"])
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.getvalue())
    name = f"scanned_{source}_{grade}"
    out.save(OUT / f"{name}.pdf")
    truth[name] = {**truth[source], "note": f"{grade} damage: {spec}"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (
        grid_simple,
        borderless_financial,
        merged_header,
        multipage,
        wide,
        rowspan,
        text_only,
        charts,
    ):
        fn()
    for src in ("grid_simple", "borderless_financial", "merged_header"):
        scanned(src)
    for src in ("grid_simple", "borderless_financial", "merged_header", "wide"):
        for grade in DAMAGE:
            degraded(src, grade)
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print(f"wrote {len(truth)} documents to {OUT}")
    for name in sorted(truth):
        n_cells = sum(len(r) for t in truth[name]["tables"] for r in t)
        print(
            f"  {name:26s} tables={len(truth[name]['tables'])} cells={n_cells} labels={len(truth[name]['labels'])}"
        )


if __name__ == "__main__":
    main()
