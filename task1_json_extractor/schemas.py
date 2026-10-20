from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class InvoiceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=1)
    quantity: float = Field(gt=0)
    unit_price: float = Field(ge=0)
    total: float = Field(ge=0)


class InvoiceData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    invoice_number: str = Field(min_length=1)
    invoice_date: str = Field(min_length=1)
    seller: str = Field(min_length=1)
    customer: str = Field(min_length=1)
    items: list[InvoiceItem] = Field(min_length=1)
    currency: str = Field(min_length=1)
    payment_status: Literal["Paid", "Unpaid", "Overdue"]
    subtotal: float = Field(ge=0)
    total: float = Field(ge=0)