"""
Vector API - Pure Vector Extraction
Extracts raw vector data, text, and geometric elements from PDF files
No business logic - only data extraction for other services to process
"""
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Query
import logging
import re
from typing import List, Dict, Any, Optional
import math
from pydantic import BaseModel
import time
from datetime import datetime
import os
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vector_api")

# Constants for Vector Extraction
MAX_PAGE_SIZE = 10000  # Max dimension for very large pages
MIN_LINE_LENGTH = 0.1  # Minimum line length to consider (in points)
TEXT_BLOCKS_METHOD = "dict"  # Use 'dict' for better text block extraction

# Utility functions
def point_to_dict(p) -> dict:
    """Convert PyMuPDF point to dictionary with 2 decimal places precision"""
    return {"x": round(p.x, 2), "y": round(p.y, 2)}

def rect_to_dict(r) -> dict:
    """Convert PyMuPDF rectangle to dictionary with 2 decimal places precision"""
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
    """
    Determine if a page contains vector content
    Used for vectorcheck as per the knowledge base
    """
    # Check for vector drawings
    drawings = page.get_drawings()
    if drawings:
        return True
    
    # Check for vector text
    text_dict = page.get_text("dict")
    if text_dict and "blocks" in text_dict and text_dict["blocks"]:
        return True
    
    return False

app = FastAPI(
    title="Vector Extraction API",
    description="Extracts raw vector data, text, and geometric elements from PDF files",
    version="4.0.0",
)

class MemoryMonitor:
    """Monitor memory usage"""
    @staticmethod
    def get_memory_usage():
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0  # psutil not available

class VectorCheckResponse(BaseModel):
    """Response model for vector check endpoint"""
    page_number: int
    is_vector: bool
    page_url: Optional[str] = None

@app.post("/extract-vectors/")
async def extract_vectors(file: UploadFile = File(...)):
    """
    Extract raw vector data and text from PDF file
    No business logic - only data extraction
    
    Returns:
        JSON with raw page data containing:
        - drawings: vector paths, lines, rectangles, curves
        - texts: text blocks with position and font info
        - page_info: page dimensions and metadata
    """
    start_time = time.time()
    
    try:
        logger.info(f"Extracting vectors from: {file.filename} [{MemoryMonitor.get_memory_usage():.1f} MB]")
        
        # Read PDF file
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Empty PDF file")
            
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(pdf_document) == 0:
            raise ValueError("PDF contains no pages")
            
        logger.info(f"PDF loaded: {len(pdf_document)} pages, {len(pdf_bytes)/1024:.1f} KB")
        
        output = {
            "pages": [],
            "summary": {
                "total_pages": len(pdf_document),
                "total_lines": 0,
                "total_rectangles": 0,
                "total_curves": 0,
                "total_texts": 0,
                "file_size_mb": round(len(pdf_bytes) / (1024 * 1024), 2),
                "processing_time_ms": 0
            }
        }
        
        for page_num, page in enumerate(pdf_document):
            page_start_time = time.time()
            logger.info(f"Processing page {page_num + 1} of {len(pdf_document)}")
            
            # Handle very large pages by extracting at a reasonable resolution
            if page.rect.width > MAX_PAGE_SIZE or page.rect.height > MAX_PAGE_SIZE:
                logger.warning(f"Very large page detected: {page.rect.width}x{page.rect.height}")
                # Consider downscaling for extremely large pages
            
            # Check if page contains vector content
            has_vector = is_vector_page(page)
            logger.info(f"Page {page_num + 1} vector content: {has_vector}")
            
            # Extract all geometric elements
            lines = []
            rectangles = []
            curves = []
            
            # Extract vector drawings
            for path in page.get_drawings():
                for item in path["items"]:
                    item_type = item[0]
                    points = item[1:]
                    
                    if item_type == "l":  # Line
                        p1 = point_to_dict(points[0])
                        p2 = point_to_dict(points[1])
                        line_length = distance(p1, p2)
                        
                        # Skip very short lines that are likely noise
                        if line_length < MIN_LINE_LENGTH:
                            continue
                            
                        line_data = {
                            "type": "line", 
                            "p1": p1, 
                            "p2": p2,
                            "length": line_length,
                            "color": path.get("color", [0, 0, 0]),
                            "width": round(path.get("width", 1), 2)
                        }
                        lines.append(line_data)
                        
                    elif item_type == "re":  # Rectangle
                        rect = rect_to_dict(points[0])
                        rect_area = rect["width"] * rect["height"]
                        
                        # Skip very small rectangles that are likely noise
                        if rect_area < 1:
                            continue
                            
                        rect_data = {
                            "type": "rect", 
                            "rect": rect,
                            "area": rect_area,
                            "color": path.get("color", [0, 0, 0]),
                            "fill": path.get("fill", [])
                        }
                        rectangles.append(rect_data)
                        
                    elif item_type == "c":  # Curve
                        curve_data = {
                            "type": "curve", 
                            "p1": point_to_dict(points[0]), 
                            "p2": point_to_dict(points[1]), 
                            "p3": point_to_dict(points[2]),
                            "color": path.get("color", [0, 0, 0]),
                            "width": round(path.get("width", 1), 2)
                        }
                        curves.append(curve_data)
            
            # Extract text blocks
            texts = []
            for text_block in page.get_text(TEXT_BLOCKS_METHOD)["blocks"]:
                if "lines" in text_block:
                    for line in text_block["lines"]:
                        for span in line["spans"]:
                            text_data = {
                                "text": span["text"],
                                "position": {
                                    "x": span["origin"][0],
                                    "y": span["origin"][1]
                                },
                                "font_size": span["size"],
                                "font_name": span["font"],
                                "color": span.get("color", [0, 0, 0]),
                                "bbox": rect_to_dict(span["bbox"])
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
                "is_vector": has_vector,
                "processing_time_ms": int((time.time() - page_start_time) * 1000)
            }
            
            output["pages"].append(page_data)
            
            # Update summary
            output["summary"]["total_lines"] += len(lines)
            output["summary"]["total_rectangles"] += len(rectangles)
            output["summary"]["total_curves"] += len(curves)
            output["summary"]["total_texts"] += len(texts)
            
            logger.info(f"Page {page_num + 1} processed: {len(lines)} lines, {len(rectangles)} rectangles, "
                       f"{len(texts)} texts [{MemoryMonitor.get_memory_usage():.1f} MB]")
        
        # Calculate total processing time
        output["summary"]["processing_time_ms"] = int((time.time() - start_time) * 1000)
        
        logger.info(f"Vector extraction completed in {output['summary']['processing_time_ms']}ms: "
                   f"{output['summary']['total_lines']} lines, {output['summary']['total_texts']} texts")
        return output
        
    except Exception as e:
        logger.error(f"Error during vector extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector-check/", response_model=VectorCheckResponse)
async def vector_check(file: UploadFile = File(...), page: int = Query(1, ge=1)):
    """
    Check if a specific page in a PDF contains vector content
    Used for vectorcheck as per the knowledge base
    
    Args:
        file: PDF file to check
        page: Page number to check (1-based)
        
    Returns:
        JSON with vector check result
    """
    try:
        logger.info(f"Checking vector content in {file.filename}, page {page}")
        
        # Read PDF file
        pdf_bytes = await file.read()
        if not pdf_bytes:
            raise ValueError("Empty PDF file")
            
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if len(pdf_document) == 0:
            raise ValueError("PDF contains no pages")
            
        if page > len(pdf_document):
            raise ValueError(f"Page {page} does not exist in the document (total pages: {len(pdf_document)})")
        
        # Get the specified page (0-based index)
        pdf_page = pdf_document[page-1]
        
        # Check if page contains vector content
        has_vector = is_vector_page(pdf_page)
        
        return {
            "page_number": page,
            "is_vector": has_vector,
            "page_url": None  # Can be populated by n8n workflow
        }
        
    except Exception as e:
        logger.error(f"Error during vector check: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vector Extraction API - Pure Data Extraction",
        "version": "4.0.0",
        "endpoints": {
            "/extract-vectors/": "Extract raw vector data from PDF",
            "/vector-check/": "Check if a page contains vector content",
            "/health/": "Health check"
        },
        "knowledge_base": "KENNISBANK BOUWTEKENING-ANALYSE V1.0 - Section 3: VECTORCHECK"
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    memory_usage = MemoryMonitor.get_memory_usage()
    return {
        "status": "healthy", 
        "service": "vector_api",
        "memory_usage_mb": round(memory_usage, 1),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 