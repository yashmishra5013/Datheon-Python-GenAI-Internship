import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from schemas import InvoiceData

load_dotenv()


SYSTEM_PROMPT = """
You are an accurate invoice data extraction assistant.
Extract information only from the supplied invoice text.
Never invent missing values. If a required field is missing,
use a clear placeholder only when the schema permits it.
Calculate item totals and invoice totals carefully.
Return data matching the supplied JSON schema.
"""


FEW_SHOT_EXAMPLE = """
Example input:
Invoice No: INV-101
Date: 2026-10-15
Seller: ABC Technologies
Customer: Rahul Sharma
Item: Laptop, Quantity: 1, Unit Price: 55000
Currency: INR
Payment Status: Unpaid

Example interpretation:
The item total is 55000. The invoice has one item.
Use the provided currency and payment status.
"""


def build_prompt(invoice_text: str) -> str:
    """Build a reusable prompt for invoice extraction."""
    return f"""
{FEW_SHOT_EXAMPLE}

Extract the invoice data from the text below.

INVOICE TEXT:
{invoice_text}
"""


def extract_invoice(invoice_text: str) -> InvoiceData:
    """Extract invoice information and validate it using Pydantic."""
    if not invoice_text.strip():
        raise ValueError("Invoice text cannot be empty.")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to the root .env file."
        )

    client = OpenAI(api_key=api_key)

    response = client.responses.parse(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(invoice_text)},
        ],
        text_format=InvoiceData,
    )

    result = response.output_parsed
    if result is None:
        raise ValueError("The model did not return valid invoice data.")

    return result


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: python main.py sample_invoice.txt",
            file=sys.stderr,
        )
        raise SystemExit(2)

    invoice_path = Path(sys.argv[1])

    if not invoice_path.is_file():
        print(f"File not found: {invoice_path}", file=sys.stderr)
        raise SystemExit(1)

    invoice_text = invoice_path.read_text(encoding="utf-8")

    try:
        invoice = extract_invoice(invoice_text)
        print(invoice.model_dump_json(indent=2))
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()