import os
import pytest
from src.db import init_db, get_connection, save_invoice, log_execution
from src.models import ExtractedInvoice
from src.extractor import build_validated_record
from app import get_invoices_df, get_logs_df, get_pending_review_files


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    # Isolates tests in a temporary SQLite database
    db_file = str(tmp_path / "test_app.db")
    monkeypatch.setenv("DATABASE_PATH", db_file)
    init_db(db_file)
    return db_file


def test_get_invoices_df_empty(test_db):
    # Test 1: Empty database returns empty DataFrame with expected columns
    df = get_invoices_df()
    assert df.empty
    expected_cols = [
        "id", "vendor_name", "invoice_number", "invoice_date",
        "due_date", "amount", "currency", "source_filename", "created_at"
    ]
    for col in expected_cols:
        assert col in df.columns


def test_get_invoices_df_with_data(test_db):
    # Test 2: Persisted invoice is queried and returned in DataFrame
    raw = ExtractedInvoice(
        vendor_name="Acme Corp",
        invoice_number="INV-1001",
        invoice_date="2024-05-01",
        due_date="2024-06-01",
        amount="2500.00",
        currency="USD",
    )
    record = build_validated_record(raw, "invoice_sample.pdf")
    save_invoice(record)

    df = get_invoices_df()
    assert len(df) == 1
    assert df.iloc[0]["vendor_name"] == "Acme Corp"
    assert df.iloc[0]["invoice_number"] == "INV-1001"
    assert df.iloc[0]["source_filename"] == "invoice_sample.pdf"


def test_get_logs_df(test_db):
    # Test 3: Execution logs are retrieved properly with details column
    log_execution(
        "invoice_test.pdf",
        "extraction",
        "success",
        "Extracted 5 fields successfully",
    )
    df = get_logs_df()
    assert len(df) == 1
    assert df.iloc[0]["source_filename"] == "invoice_test.pdf"
    assert df.iloc[0]["step"] == "extraction"
    assert df.iloc[0]["status"] == "success"
    assert df.iloc[0]["details"] == "Extracted 5 fields successfully"


def test_get_pending_review_files(tmp_path, monkeypatch):
    # Test 4: Discovers PDF/images and filters out other file extensions
    review_dir = tmp_path / "human_review"
    review_dir.mkdir()
    monkeypatch.setattr("app.HUMAN_REVIEW_DIR", str(review_dir))

    # Create dummy files
    (review_dir / "inv1.pdf").write_text("dummy")
    (review_dir / "inv2.png").write_text("dummy")
    (review_dir / "notes.txt").write_text("dummy")

    files = get_pending_review_files()
    assert sorted(files) == ["inv1.pdf", "inv2.png"]
