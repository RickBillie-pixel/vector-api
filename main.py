"""
Vector API - Advanced Construction Drawing Analysis
Extracts comprehensive vector data, text, dimensions, and construction elements from PDF files
Optimized for complex construction drawings and technical plans
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

def distance(p1: dict, p2: dict) -> float:
    """Calculate distance between two points"""
    return math.sqrt((p2['x'] - p1['x'])**2 + (p2['y'] - p1['y'])**2)

def is_dimension_text(text: str) -> bool:
    """Check if text represents a dimension"""
    # Match patterns like: 3.50, 3,50, 3.5m, 3500, etc.
    dimension_patterns = [
        r'^\d+[,.]?\d*\s*(m|cm|mm|ft|in)?$',  # 3.50, 3,50, 3.5m
        r'^\d+[,.]?\d*\s*[xX×]\s*\d+[,.]?\d*',  # 3.50 x 4.20
        r'^\d+[,.]?\d*\s*[xX×]\s*\d+[,.]?\d*\s*(m|cm|mm|ft|in)?$',  # 3.50 x 4.20m
        r'^\d+[,.]?\d*\s*[xX×]\s*\d+[,.]?\d*\s*[xX×]\s*\d+[,.]?\d*',  # 3.50 x 4.20 x 2.80
    ]
    return any(re.match(pattern, text.strip()) for pattern in dimension_patterns)

def is_room_label(text: str) -> bool:
    """Check if text is a room label"""
    room_patterns = [
        r'^(slaapkamer|bedroom|kamer|room|woonkamer|living|keuken|kitchen|badkamer|bathroom|toilet|wc|gang|hall|berging|storage|garage|kelder|basement|zolder|attic)',
        r'^(SK|BK|WK|KK|BK|T|G|B|GA|K|Z)\d*$',  # Dutch abbreviations
        r'^(BR|LR|KT|BA|WC|H|S|GA|BS|AT)\d*$',  # English abbreviations
    ]
    return any(re.search(pattern, text.strip().lower()) for pattern in room_patterns)

def is_construction_note(text: str) -> bool:
    """Check if text is a construction note"""
    note_patterns = [
        r'^(beton|concrete|hout|wood|staal|steel|metselwerk|masonry|isolatie|insulation)',
        r'^(fundering|foundation|vloer|floor|dak|roof|muur|wall|deur|door|raam|window)',
        r'^(elektra|electrical|sanitair|plumbing|ventilatie|ventilation|verwarming|heating)',
        r'^(afwerking|finishing|tegels|tiles|verf|paint|behang|wallpaper)',
    ]
    return any(re.search(pattern, text.strip().lower()) for pattern in note_patterns)

def extract_dimensions_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract dimension information from text"""
    # Match dimension patterns
    dimension_match = re.search(r'(\d+[,.]?\d*)\s*[xX×]\s*(\d+[,.]?\d*)(?:\s*[xX×]\s*(\d+[,.]?\d*))?\s*(m|cm|mm|ft|in)?', text)
    if dimension_match:
        width = float(dimension_match.group(1).replace(',', '.'))
        height = float(dimension_match.group(2).replace(',', '.'))
        depth = float(dimension_match.group(3).replace(',', '.')) if dimension_match.group(3) else None
        unit = dimension_match.group(4) or 'm'
        
        return {
            "width": width,
            "height": height,
            "depth": depth,
            "unit": unit,
            "area": width * height if depth is None else width * height * depth
        }
    return None

app = FastAPI(
    title="Advanced Vector Extraction API",
    description="Extracts comprehensive construction data from PDF files including dimensions, annotations, and technical details",
    version="2.0.0",
)

@app.post("/extract-vectors/")
async def extract_vectors(file: UploadFile = File(...)):
    """
    Extract comprehensive vector data and construction information from PDF file
    
    Returns:
        JSON with detailed page data containing:
        - drawings: vector paths, lines, rectangles, curves with construction analysis
        - texts: text blocks with semantic classification
        - dimensions: extracted dimension information
        - annotations: construction notes and specifications
        - room_labels: identified room labels
        - construction_elements: detected building elements
    """
    try:
        logger.info(f"Processing construction drawing: {file.filename}")
        
        # Read PDF file
        pdf_bytes = await file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        output = {
            "pages": [],
            "summary": {
                "total_pages": len(pdf_document),
                "total_drawings": 0,
                "total_texts": 0,
                "total_dimensions": 0,
                "total_room_labels": 0,
                "total_annotations": 0,
                "file_size_mb": round(len(pdf_bytes) / (1024 * 1024), 2)
            }
        }
        
        for page_num, page in enumerate(pdf_document):
            logger.info(f"Processing page {page_num + 1}")
            
            page_data = {
                "page_number": page_num + 1,
                "page_size": {
                    "width": round(page.rect.width, 2),
                    "height": round(page.rect.height, 2)
                },
                "drawings": [],
                "texts": [],
                "dimensions": [],
                "room_labels": [],
                "annotations": [],
                "construction_elements": []
            }
            
            # Extract vector drawings with construction analysis
            for path in page.get_drawings():
                drawing_info = {
                    "rect": rect_to_dict(path["rect"]),
                    "color": path["color"],
                    "fill": path["fill"],
                    "width": round(path.get("width", 0), 2),
                    "items": [],
                    "construction_type": "unknown"
                }
                
                # Analyze drawing items for construction elements
                for item in path["items"]:
                    item_type = item[0]
                    points = item[1:]
                    
                    if item_type == "l":  # Line
                        line_data = {
                            "type": "line", 
                            "p1": point_to_dict(points[0]), 
                            "p2": point_to_dict(points[1]),
                            "length": distance(point_to_dict(points[0]), point_to_dict(points[1]))
                        }
                        
                        # Classify line as potential wall, dimension line, etc.
                        if line_data["length"] > 50:  # Long lines are likely walls
                            line_data["construction_type"] = "potential_wall"
                        elif line_data["length"] < 20:  # Short lines might be dimension lines
                            line_data["construction_type"] = "potential_dimension"
                        
                        drawing_info["items"].append(line_data)
                        
                    elif item_type == "re":  # Rectangle
                        rect_data = {
                            "type": "rect", 
                            "rect": rect_to_dict(points[0]),
                            "area": rect_to_dict(points[0])["width"] * rect_to_dict(points[0])["height"]
                        }
                        
                        # Classify rectangles as rooms, doors, windows, etc.
                        if rect_data["area"] > 10000:  # Large rectangles are likely rooms
                            rect_data["construction_type"] = "potential_room"
                        elif rect_data["area"] < 1000:  # Small rectangles might be doors/windows
                            rect_data["construction_type"] = "potential_opening"
                        
                        drawing_info["items"].append(rect_data)
                        
                    elif item_type == "c":  # Curve
                        curve_data = {
                            "type": "curve", 
                            "p1": point_to_dict(points[0]), 
                            "p2": point_to_dict(points[1]), 
                            "p3": point_to_dict(points[2])
                        }
                        drawing_info["items"].append(curve_data)
                
                # Determine overall construction type
                if any(item.get("construction_type") == "potential_wall" for item in drawing_info["items"]):
                    drawing_info["construction_type"] = "wall_structure"
                elif any(item.get("construction_type") == "potential_room" for item in drawing_info["items"]):
                    drawing_info["construction_type"] = "room_boundary"
                elif any(item.get("construction_type") == "potential_opening" for item in drawing_info["items"]):
                    drawing_info["construction_type"] = "opening"
                
                page_data["drawings"].append(drawing_info)
            
            # Extract and classify text blocks
            for block in page.get_text("dict")["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_data = {
                                "text": span["text"].strip(),
                                "font": span["font"],
                                "size": round(span["size"], 2),
                                "color": span["color"],
                                "bbox": rect_to_dict(span["bbox"]),
                                "text_type": "general"
                            }
                            
                            # Classify text based on content
                            if is_dimension_text(span["text"]):
                                text_data["text_type"] = "dimension"
                                page_data["dimensions"].append(text_data)
                                
                                # Extract dimension data
                                dim_data = extract_dimensions_from_text(span["text"])
                                if dim_data:
                                    text_data["dimension_data"] = dim_data
                                    
                            elif is_room_label(span["text"]):
                                text_data["text_type"] = "room_label"
                                page_data["room_labels"].append(text_data)
                                
                            elif is_construction_note(span["text"]):
                                text_data["text_type"] = "construction_note"
                                page_data["annotations"].append(text_data)
                                
                            page_data["texts"].append(text_data)
            
            # Update summary counts
            output["summary"]["total_drawings"] += len(page_data["drawings"])
            output["summary"]["total_texts"] += len(page_data["texts"])
            output["summary"]["total_dimensions"] += len(page_data["dimensions"])
            output["summary"]["total_room_labels"] += len(page_data["room_labels"])
            output["summary"]["total_annotations"] += len(page_data["annotations"])
            
            output["pages"].append(page_data)
        
        logger.info(f"Successfully extracted construction data from {len(output['pages'])} pages")
        logger.info(f"Summary: {output['summary']}")
        return output
        
    except Exception as e:
        logger.error(f"Error processing construction drawing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vector-api", "version": "2.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 