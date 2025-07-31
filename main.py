from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List
import httpx
import pypdf
import io
import google.generativeai as genai
import os
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY", "your-hackrx-api-key-here")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")

# Security
security = HTTPBearer()

# Request/Response Models
class HackRXRequest(BaseModel):
    documents: str  # PDF URL
    questions: List[str]

class HackRXResponse(BaseModel):
    answers: List[str]

# Startup/Shutdown Events
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting HackRX API Server...")
    yield
    logger.info("Shutting down HackRX API Server...")

# Initialize FastAPI
app = FastAPI(
    title="HackRX Document Q&A API",
    description="AI-powered document analysis using Gemini 2.5 Pro",
    version="1.0.0",
    lifespan=lifespan
)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the Bearer token"""
    if credentials.credentials != HACKRX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

async def download_pdf(url: str) -> bytes:
    """Download PDF from URL"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content
    except Exception as e:
        logger.error(f"Error downloading PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {str(e)}")

def extract_text_from_pdf(pdf_content: bytes) -> str:
    """Extract text from PDF content"""
    try:
        pdf_file = io.BytesIO(pdf_content)
        pdf_reader = pypdf.PdfReader(pdf_file)
        
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            except Exception as e:
                logger.warning(f"Error extracting text from page {page_num + 1}: {str(e)}")
                continue
        
        if not text.strip():
            raise Exception("No text could be extracted from the PDF")
            
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to extract text from PDF: {str(e)}")

async def answer_questions_with_gemini(document_text: str, questions: List[str]) -> List[str]:
    """Use Gemini 2.5 Pro to answer questions based on document text"""
    try:
        if not GEMINI_API_KEY:
            raise Exception("Gemini API key not configured")
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        answers = []
        
        for question in questions:
            try:
                # Create a comprehensive prompt
                prompt = f"""
You are an expert document analyst. Based on the following document content, please answer the question accurately and concisely.

DOCUMENT CONTENT:
{document_text}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the information provided in the document
2. If the information is not found in the document, say "Information not found in the document"
3. Be precise and direct in your answer
4. Include specific details, numbers, or timeframes when mentioned in the document
5. Keep the answer concise but complete

ANSWER:"""

                # Generate response
                response = model.generate_content(prompt)
                answer = response.text.strip()
                answers.append(answer)
                
                logger.info(f"Question: {question[:50]}... | Answer generated successfully")
                
            except Exception as e:
                logger.error(f"Error answering question '{question}': {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        return answers
        
    except Exception as e:
        logger.error(f"Error with Gemini API: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI processing error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "HackRX Document Q&A API is running!",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "gemini_configured": bool(GEMINI_API_KEY),
        "api_key_configured": bool(HACKRX_API_KEY)
    }

@app.post("/hackrx/run", response_model=HackRXResponse)
async def process_document_questions(
    request: HackRXRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint to process document and answer questions
    """
    try:
        logger.info(f"Processing request with {len(request.questions)} questions")
        logger.info(f"Document URL: {request.documents}")
        
        # Step 1: Download PDF
        logger.info("Downloading PDF...")
        pdf_content = await download_pdf(request.documents)
        logger.info(f"PDF downloaded successfully, size: {len(pdf_content)} bytes")
        
        # Step 2: Extract text from PDF
        logger.info("Extracting text from PDF...")
        document_text = extract_text_from_pdf(pdf_content)
        logger.info(f"Text extracted successfully, length: {len(document_text)} characters")
        
        # Step 3: Answer questions using Gemini
        logger.info("Generating answers with Gemini 2.5 Pro...")
        answers = await answer_questions_with_gemini(document_text, request.questions)
        logger.info(f"Generated {len(answers)} answers successfully")
        
        return HackRXResponse(answers=answers)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return {"error": "Endpoint not found", "detail": "Check your URL path"}

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return {"error": "Internal server error", "detail": "Something went wrong on our end"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )