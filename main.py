"""
Vector API - Pure Vector Extraction
Extracts raw vector data, text, and geometric elements from PDF files
No business logic - only data extraction for other services to process
"""
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException
import logging
import re
from typing import List, Dict, Any, Optional
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_api")

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

app = FastAPI(
    title="Vector Extraction API",
    description="Extracts raw vector data, text, and geometric elements from PDF files",
    version="4.0.0",
)

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
    try:
        logger.info(f"Extracting vectors from: {file.filename}")
        
        # Read PDF file
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        output = {
            "pages": [],
            "summary": {
                "total_pages": len(pdf_document),
                "total_lines": 0,
                "total_rectangles": 0,
                "total_curves": 0,
                "total_texts": 0,
                "file_size_mb": round(len(pdf_bytes) / (1024 * 1024), 2)
            }
        }
        
        for page_num, page in enumerate(pdf_document):
            logger.info(f"Processing page {page_num + 1}")
            
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
                        line_data = {
                            "type": "line", 
                            "p1": point_to_dict(points[0]), 
                            "p2": point_to_dict(points[1]),
                            "length": distance(point_to_dict(points[0]), point_to_dict(points[1])),
                            "color": path.get("color", [0, 0, 0]),
                            "width": round(path.get("width", 1), 2)
                        }
                        lines.append(line_data)
                        
                    elif item_type == "re":  # Rectangle
                        rect_data = {
                            "type": "rect", 
                            "rect": rect_to_dict(points[0]),
                            "area": rect_to_dict(points[0])["width"] * rect_to_dict(points[0])["height"],
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
            for text_block in page.get_text("dict")["blocks"]:
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
                "texts": texts
            }
            
            output["pages"].append(page_data)
            
            # Update summary
            output["summary"]["total_lines"] += len(lines)
            output["summary"]["total_rectangles"] += len(rectangles)
            output["summary"]["total_curves"] += len(curves)
            output["summary"]["total_texts"] += len(texts)
        
        logger.info(f"Vector extraction completed: {output['summary']['total_lines']} lines, {output['summary']['total_texts']} texts")
        return output
        
    except Exception as e:
        logger.error(f"Error during vector extraction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Vector Extraction API - Pure Data Extraction",
        "version": "4.0.0",
        "endpoints": {
            "/extract-vectors/": "Extract raw vector data from PDF",
            "/health/": "Health check"
        }
    }

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vector_api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 