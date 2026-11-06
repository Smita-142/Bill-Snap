# Bill Snap - OCR & Bill Classification Engine

Bill Snap features a high-performance OCR pipeline powered by **PaddleOCR** with automatic bill vs non-bill classification, regex heuristic field extraction, and multiprocess batch folder processing.

---

## 1. Setup & Installation

Ensure you have Python 3.11+ installed.

```bash
# Navigate to backend directory
cd backend

# Activate your virtual environment
# Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Note on Windows CPU Execution**: PaddleX requires setting `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False` to ensure optimal and bug-free execution on x86 Windows platforms without oneDNN instruction incompatibilities. This is preconfigured in `ocr_service.py` and `batch_ocr.py`.

---

## 2. Configuration (`backend/app/ml/ocr_config.yaml`)

OCR and classification parameters can be tuned in `backend/app/ml/ocr_config.yaml`:

```yaml
ocr:
  engine: "paddleocr"
  language: "en"          # Primary OCR language ('en', 'hi', etc.)
  use_gpu: false          # Set to true for CUDA GPU acceleration
  use_angle_cls: true     # Orientation and textline angle classification
  max_long_side: 1800     # Resizes longest side (never upscales)
  workers: 2              # Number of parallel worker processes

classification:
  threshold: 0.35         # Score threshold for bill vs non-bill decision
  keywords:
    - "tax invoice"
    - "invoice no"
    - "bill no"
    - "grand total"
    - "subtotal"
    - "invoice"
    - "receipt"
    - "bill"
    - "total"
    - "gst"
    - "cgst"
    - "sgst"
    - "igst"
    - "tax"
    - "amount"
    - "qty"
    - "rate"
    - "cash"
    - "paid"
    - "due"
  weights:
    keyword: 1.0
    currency: 1.5         # ₹, Rs, INR, $
    date: 1.0             # DD/MM/YYYY, YYYY-MM-DD, etc.
    gstin: 2.0            # 15-character GSTIN format
```

---

## 3. Batch / Folder Processing

Process an entire folder of images in parallel with `concurrent.futures.ProcessPoolExecutor`:

```bash
# Run batch OCR from backend directory:
python -m app.ml.batch_ocr --folder path/to/images --output path/to/output --workers 2
```

### Outputs Generated:
- **Per-image timing logs**: Real-time progress in stdout
- **Batch summary table**: Total processed, bills detected, non-bills, average time
- **`results.json`**: Structured JSON array per image:
  ```json
  {
    "file": "path/to/receipt.jpg",
    "is_bill": true,
    "score": 1.0,
    "matched_signals": ["keyword:invoice", "keyword:total", "pattern:currency"],
    "fields": {
      "vendor_name": "Fresh Mart",
      "bill_date": "15/10/2026",
      "bill_number": "INV-2026-9812",
      "total_amount": 22.77
    },
    "text": "Extracted OCR text...",
    "processing_time_sec": 1.84
  }
  ```
- **`results.csv`**: Tabular export with columns: `file`, `is_bill`, `score`, `vendor_name`, `bill_date`, `bill_number`, `total_amount`, `processing_time_sec`, `matched_signals`.

---

## 4. Testing & Verification

Run the comprehensive test suite and benchmark:

```bash
# Run backend pytest suite (68 tests)
pytest backend/tests/ -v

# Run PaddleOCR batch benchmark & verification script
python backend/tests/test_paddle_batch.py
```

