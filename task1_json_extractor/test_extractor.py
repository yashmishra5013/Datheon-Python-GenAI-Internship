import pytest
from pydantic import ValidationError

from main import build_prompt, extract_invoice
from schemas import InvoiceData, InvoiceItem


def test_prompt_contains_invoice_text():
    prompt = build_prompt("Seller: ABC Technologies")
    assert "ABC Technologies" in prompt
    assert "INVOICE TEXT" in prompt


def test_empty_invoice_is_rejected():
    with pytest.raises(ValueError, match="cannot be empty"):
        extract_invoice("   ")


def test_valid_invoice_schema():
    invoice = InvoiceData(
        invoice_number="INV-101",
        invoice_date="2026-10-15",
        seller="ABC Technologies",
        customer="Rahul Sharma",
        items=[
            InvoiceItem(
                description="Laptop",
                quantity=1,
                unit_price=55000,
                total=55000,
            )
        ],
        currency="INR",
        payment_status="Unpaid",
        subtotal=55000,
        total=55000,
    )

    assert invoice.total == 55000
    assert invoice.items[0].description == "Laptop"


def test_invalid_payment_status_is_rejected():
    with pytest.raises(ValidationError):
        InvoiceData(
            invoice_number="INV-101",
            invoice_date="2026-10-15",
            seller="ABC Technologies",
            customer="Rahul Sharma",
            items=[
                InvoiceItem(
                    description="Laptop",
                    quantity=1,
                    unit_price=55000,
                    total=55000,
                )
            ],
            currency="INR",
            payment_status="Pending",
            subtotal=55000,
            total=55000,
        )


def test_negative_quantity_is_rejected():
    with pytest.raises(ValidationError):
        InvoiceItem(
            description="Laptop",
            quantity=-1,
            unit_price=55000,
            total=55000,
        )