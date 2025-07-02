"""
Vector API - Extracts raw vectors and text from PDF files
This is the only service that directly reads PDF files
"""
import fitz  # PyMuPDF
from fastapi import FastAPI, File, UploadFile, HTTPException
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vector_api")

# Utility functions (inline to avoid shared module dependency)
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

app = FastAPI(
    title="Vector Extraction API",
    description="Extracts raw vectors and text from PDF files",
    version="1.0.0",
)

@app.post("/extract-vectors/")
async def extract_vectors(file: UploadFile = File(...)):
    """
    Extract raw vectors and text from PDF file
    
    Returns:
        JSON with page data containing:
        - drawings: vector paths, lines, rectangles, curves
        - texts: text blocks with position and styling
    """
    try:
        logger.info(f"Processing PDF file: {file.filename}")
        
        # Read PDF file
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        output = {"pages": []}
        
        for page_num, page in enumerate(pdf_document):
            logger.info(f"Processing page {page_num + 1}")
            
            page_data = {
                "page_number": page_num + 1,
                "drawings": [],
                "texts": []
            }
            
            # Extract vector drawings
            for path in page.get_drawings():
                drawing_info = {
                    "rect": rect_to_dict(path["rect"]),
                    "color": path["color"],
                    "fill": path["fill"],
                    "width": round(path.get("width", 0), 2),
                    "items": []
                }
                
                for item in path["items"]:
                    item_type = item[0]
                    points = item[1:]
                    
                    if item_type == "l":  # Line
                        drawing_info["items"].append({
                            "type": "line", 
                            "p1": point_to_dict(points[0]), 
                            "p2": point_to_dict(points[1])
                        })
                    elif item_type == "re":  # Rectangle
                        drawing_info["items"].append({
                            "type": "rect", 
                            "rect": rect_to_dict(points[0])
                        })
                    elif item_type == "c":  # Curve
                        drawing_info["items"].append({
                            "type": "curve", 
                            "p1": point_to_dict(points[0]), 
                            "p2": point_to_dict(points[1]), 
                            "p3": point_to_dict(points[2])
                        })
                
                page_data["drawings"].append(drawing_info)
            
            # Extract text blocks
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_data["texts"].append({
                                "text": span["text"],
                                "font": span["font"],
                                "size": round(span["size"], 2),
                                "color": span["color"],
                                "bbox": rect_to_dict(span["bbox"])
                            })
            
            output["pages"].append(page_data)
        
        logger.info(f"Successfully extracted data from {len(output['pages'])} pages")
        return output
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vector-api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 