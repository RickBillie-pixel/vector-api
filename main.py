"""
Vector Extraction API - Extracts raw vector data, text, and viewports from PDF files or URLs
Minimal version for Render.com deployment and n8n integration
"""

import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import logging
import math
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
import requests
from shapely.geometry import box, Point
import numpy as np
from sklearn.cluster import DBSCAN

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger("vector_api")

# Constants
MAX_PAGE_SIZE = 10000
MIN_LINE_LENGTH = 0.1
TEXT_BLOCKS_METHOD = "dict"
VIEWPORT_KEYWORDS = ["detail", "section", "legenda", "titelblok", "title block"]

# Utility functions
def point_to_dict(p) -> dict:
    """Convert PyMuPDF point to dictionary"""
    return {"x": round(p.x, 2), "y": round(p.y, 2)}

def rect_to_dict(r) -> dict:
    """Convert PyMuPDF rectangle to dictionary"""
    if isinstance(r, tuple):
        return {
            "x0": round(r[0], 2), "y0": round(r[1], 2),
            "x1": round(r[2], 2), "y1": round(r[3], 2),
            "width": round(r[2] - r[0], 2), "height": round(r[3] - r[1], 2)
        }
    return {
        "x0": round(r.x0, 2), "y0": round(r.y0, 2),
        "x1": round(r.x1, 2), "y1": round(r.y1, 2),
        "width": round(r.width, 2), "height": round(r.height, 2)
    }

def distance(p1: dict, p2: dict) -> float:
    """Calculate distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_vector_page(page) -> bool:
    """Determine if a page contains vector content"""
    drawings = page.get_drawings()
    if drawings:
        return True
    text_dict = page.get_text("dict")
    if text_dict and "blocks" in text_dict and text_dict["blocks"]:
        return True
    return False

def detect_viewports(page, text_blocks: List[Dict]) -> List[Dict[str, Any]]:
    """Detect viewports using text labels and clustering"""
    try:
        viewports = []
        text_positions = [
            (span["origin"][0], span["origin"][1], span["text"].strip().lower())
            for block in text_blocks if "lines" in block
            for line in block["lines"] for span in line["spans"]
            if any(kw in span["text"].strip().lower() for kw in VIEWPORT_KEYWORDS)
        ]

        if not text_positions:
            return []

        if len(text_positions) > 1:
            X = np.array([(x, y) for x, y, _ in text_positions])
            dbscan = DBSCAN(eps=50.0, min_samples=1).fit(X)
            labels = dbscan.labels_
            for idx, (x, y, text) in enumerate(text_positions):
                if labels[idx] != -1:
                    bbox = box(x - 50, y - 50, x + 50, y + 50)
                    viewports.append({
                        "label": text,
                        "bbox": bbox,
                        "id": f"viewport_{labels[idx]}"
                    })
        else:
            x, y, text = text_positions[0]
            bbox = box(x - 50, y - 50, x + 50, y + 50)
            viewports.append({
                "label": text,
                "bbox": bbox,
                "id": "viewport_0"
            })

        return viewports
    except Exception as e:
        logger.error(f"Error detecting viewports: {e}")
        return []

class VectorCheckResponse(BaseModel):
    page_number: int
    is_vector: bool
    page_url: Optional[str] = None

class UrlRequest(BaseModel):
    url: HttpUrl = Field(..., description="URL of the PDF file")
    page_number: int = Field(..., ge=1, description="Page number to process")

app = FastAPI(
    title="Vector Extraction API",
    description="Extracts raw vector data, text, and viewports from PDF files or URLs",
    version="4.8.0",
)

@app.exception_handler(Exception)
async def custom_exception_handler(request, exc: Exception):
    """Global exception handler for security"""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"}
    )

def extract_vectors(data: bytes, filename: str) -> Dict[str, Any]:
    """Extract raw vector data, text, and viewports from PDF"""
    start_time = time.time()

    try:
        pdf_document = fitz.open(stream=data, filetype="pdf")
        if len(pdf_document) == 0:
            raise ValueError("PDF contains no pages")

        output = {
            "pages": [],
            "summary": {
                "filename": filename,
                "total_pages": len(pdf_document),
                "total_lines": 0,
                "total_rectangles": 0,
                "total_curves": 0,
                "total_texts": 0,
                "total_viewports": 0,
                "file_size_mb": round(len(data) / (1024 * 1024), 2),
                "processing_time_ms": 0
            }
        }

        for page_num, page in enumerate(pdf_document):
            page_start_time = time.time()
            logger.info(f"Processing page {page_num + 1}")

            if page.rect.width > MAX_PAGE_SIZE or page.rect.height > MAX_PAGE_SIZE:
                logger.warning(f"Very large page detected: {page.rect.width}x{page.rect.height}")

            has_vector = is_vector_page(page)
            text_dict = page.get_text(TEXT_BLOCKS_METHOD)
            viewports = detect_viewports(page, text_dict["blocks"])

            lines, rectangles, curves = [], [], []
            for path in page.get_drawings():
                for item in path["items"]:
                    item_type = item[0]
                    points = item[1:]

                    if item_type == "l":
                        p1 = point_to_dict(points[0])
                        p2 = point_to_dict(points[1])
                        line_length = distance(p1, p2)
                        if line_length < MIN_LINE_LENGTH:
                            continue
                        viewport_id = "default"
                        for vp in viewports:
                            if Point(p1["x"], p1["y"]).within(vp["bbox"]):
                                viewport_id = vp["id"]
                                break
                        line_data = {
                            "type": "line",
                            "p1": p1,
                            "p2": p2,
                            "length": line_length,
                            "color": path.get("color", [0, 0, 0]),
                            "width": round(path.get("width", 1), 2),
                            "viewport_id": viewport_id
                        }
                        lines.append(line_data)

                    elif item_type == "re":
                        rect = rect_to_dict(points[0])
                        rect_area = rect["width"] * rect["height"]
                        if rect_area < 1:
                            continue
                        viewport_id = "default"
                        for vp in viewports:
                            if Point(rect["x0"], rect["y0"]).within(vp["bbox"]):
                                viewport_id = vp["id"]
                                break
                        rect_data = {
                            "type": "rect",
                            "rect": rect,
                            "area": rect_area,
                            "color": path.get("color", [0, 0, 0]),
                            "fill": path.get("fill", [])
                        }
                        rectangles.append(rect_data)

                    elif item_type == "c":
                        viewport_id = "default"
                        for vp in viewports:
                            if Point(points[0]["x"], points[0]["y"]).within(vp["bbox"]):
                                viewport_id = vp["id"]
                                break
                        curve_data = {
                            "type": "curve",
                            "p1": point_to_dict(points[0]),
                            "p2": point_to_dict(points[1]),
                            "p3": point_to_dict(points[2]),
                            "color": path.get("color", [0, 0, 0]),
                            "width": round(path.get("width", 1), 2),
                            "viewport_id": viewport_id
                        }
                        curves.append(curve_data)

            texts = []
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"].strip()
                            if len(text) < 2:
                                continue
                            viewport_id = "default"
                            for vp in viewports:
                                if Point(span["origin"][0], span["origin"][1]).within(vp["bbox"]):
                                    viewport_id = vp["id"]
                                    break
                            text_data = {
                                "text": text,
                                "position": {"x": span["origin"][0], "y": span["origin"][1]},
                                "font_size": span["size"],
                                "font_name": span["font"],
                                "color": span.get("color", [0, 0, 0]),
                                "bbox": rect_to_dict(span["bbox"]),
                                "viewport_id": viewport_id
                            }
                            texts.append(text_data)

            page_data = {
                "page_number": page_num + 1,
                "page_size": {
                    "width": round(page.rect.width, 2),
                    "height": round(page.rect.height, 2)
                },
                "drawings": {
                    "lines": lines,
                    "rectangles": rectangles,
                    "curves": curves
                },
                "texts": texts,
                "viewports": [{"id": vp["id"], "label": vp["label"], "bbox": rect_to_dict(vp["bbox"].bounds)} for vp in viewports],
                "is_vector": has_vector,
                "processing_time_ms": int((time.time() - page_start_time) * 1000)
            }

            output["pages"].append(page_data)
            output["summary"]["total_lines"] += len(lines)
            output["summary"]["total_rectangles"] += len(rectangles)
            output["summary"]["total_curves"] += len(curves)
            output["summary"]["total_texts"] += len(texts)
            output["summary"]["total_viewports"] += len(viewports)

            logger.info(f"Page {page_num + 1} processed: {len(lines)} lines, {len(rectangles)} rectangles, "
                       f"{len(texts)} texts, {len(viewports)} viewports")

        output["summary"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
        return output

    except Exception as e:
        logger.error(f"Error in extract_vectors: {e}", exc_info=True)
        raise

    finally:
        if "pdf_document" in locals():
            pdf_document.close()

@app.post("/extract-vectors/")
async def extract_vectors(file: UploadFile = File(...)):
    """Extract raw vector data, text, and viewports from PDF file"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        logger.info(f"Extracting vectors from: {file.filename}")
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Empty PDF file")

        return extract_vectors(data=pdf_bytes, filename=file.filename)

    except Exception as e:
        logger.error(f"Error during vector extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/extract-vectors-from-urls/", response_model=List[Dict[str, Any]])
async def extract_vectors_from_urls(urls: List[UrlRequest]):
    """Extract raw vector data, text, and viewports from PDF URLs"""
    results = []
    for url_request in urls:
        try:
            logger.info(f"Downloading PDF from: {url_request.url}")
            response = requests.get(url_request.url, timeout=10)
            response.raise_for_status()
            pdf_bytes = response.content
            if not pdf_bytes:
                logger.warning(f"Empty PDF from URL: {url_request.url}")
                continue

            pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
            if len(pdf_document) == 0:
                logger.warning(f"No pages in PDF from URL: {url_request.url}")
                continue
            if url_request.page_number > len(pdf_document):
                logger.warning(f"Page {url_request.page_number} does not exist in PDF from URL: {url_request.url}")
                continue

            output = extract_vectors(data=pdf_bytes, filename=str(url_request.url))
            output = {
                "url": str(url_request.url),
                "page_number": url_request.page_number,
                "data": next((p for p in output["pages"] if p["page_number"] == url_request.page_number), None)
            }
            if output["data"]:
                results.append(output)
            else:
                logger.warning(f"No data extracted for page {url_request.page_number} from URL: {url_request.url}")

        except requests.RequestException as e:
            logger.error(f"Error downloading PDF from {url_request.url}: {e}")
            continue

    return results if results else [{"url": "", "page_number": 0, "data": None}]

@app.post("/vector-check/", response_model=VectorCheckResponse)
async def vector_check(file: UploadFile = File(...), page: int = Query(1, ge=1)):
    """Check if a specific page contains vector content"""
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        logger.info(f"Checking vector content in {file.filename}, page {page}")
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Empty PDF file")

        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(pdf_document) == 0:
            raise ValueError("PDF contains no pages")
        if page > len(pdf_document):
            raise ValueError(f"Page {page} does not exist")

        pdf_page = pdf_document[page-1]
        has_vector = is_vector_page(pdf_page)

        return {
            "page_number": page,
            "is_vector": has_vector,
            "page_url": None
        }

    except Exception as e:
        logger.error(f"Error during vector check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vector Extraction API - Pure Data Extraction",
        "version": "4.8.0",
        "endpoints": {
            "/extract-vectors/": "Extract raw vector data and viewports from PDF file",
            "/extract-vectors-from-urls/": "Extract raw vector data and viewports from PDF URLs",
            "/vector-check/": "Check if a page contains vector content",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "vector_api",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
