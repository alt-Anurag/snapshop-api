# --- IMPORTS ---
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
import base64
from io import BytesIO
import time
import os
from dotenv import load_dotenv
import supabase

# --- LOAD ENVIRONMENT VARIABLES ---
# This ensures that SUPABASE_URL, SUPABASE_KEY, and ALLOWED_ORIGINS are loaded from .env
load_dotenv()

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="SnapShop Recommendation API",
    version="1.0.0",
    description="API for image-based product recommendations using CLIP embeddings and Supabase.",
)

# --- CORS CONFIGURATION ---
# Get allowed origins from environment variable, split by comma, or default to an empty list
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(',') if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Your Netlify frontend URL will go here
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- SUPABASE CLIENT INITIALIZATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Supabase URL and Key must be set in environment variables.")

try:
    supabase_client = supabase.create_client(SUPABASE_URL, SUPABASE_KEY)
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    # In a real app, you might want a more graceful startup failure or health check
    raise

# --- CLIP MODEL INITIALIZATION ---
# Load the SentenceTransformer model for CLIP embeddings
try:
    # Set this environment variable to prioritize CPU, saving VRAM if GPU is available
    # and preventing issues if PyTorch tries to use GPU but isn't configured for it.
    os.environ['CUDA_VISIBLE_DEVICES'] = '' # Ensure no GPU is used by PyTorch
    model = SentenceTransformer('clip-ViT-B-32')
    print("CLIP model loaded successfully.")
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    raise

# --- HELPER FUNCTIONS ---

def encode_image(image_bytes: bytes):
    """Encodes an image from bytes into a CLIP vector."""
    try:
        image = Image.open(BytesIO(image_bytes))
        # Ensure image is in RGB format for CLIP
        if image.mode != "RGB":
            image = image.convert("RGB")
        return model.encode(image).tolist()
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None

# --- Pydantic Models for Request/Response ---

class ImageRequest(BaseModel):
    # base64 string of the image (e.g., data:image/jpeg;base64,...)
    image_data: str

class ProductRecommendation(BaseModel):
    Product: str
    Image: HttpUrl # Use HttpUrl for URL validation
    URL: HttpUrl    # Use HttpUrl for URL validation
    Price: str      # Keeping as string as per your possible structure, adjust if numeric
    similarity: float

class RecommendationResponse(BaseModel):
    message: str
    recommendations: list[ProductRecommendation]
    query_image_url: str | None = None # Optional: if you want to echo the image URL

# --- API ENDPOINTS ---

@app.get("/")
async def read_root():
    """Basic health check endpoint."""
    return {"message": "SnapShop Recommendation API is running!"}

@app.post("/recommendations", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
async def get_recommendations(request: ImageRequest):
    """
    Receives a base64 image, generates its CLIP embedding,
    and fetches similar products from Supabase.
    """
    image_data_b64 = request.image_data

    # 1. Decode base64 image
    try:
        # Remove data URI scheme prefix if present (e.g., "data:image/jpeg;base64,")
        if "," in image_data_b64:
            header, base64_string = image_data_b64.split(",", 1)
        else:
            base64_string = image_data_b64
        image_bytes = base64.b64decode(base64_string)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 image data: {e}"
        )

    # 2. Encode image to CLIP embedding
    query_embedding = encode_image(image_bytes)
    if query_embedding is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not process the provided image for embedding."
        )

    # 3. Query Supabase for similar products with retry logic
    match_threshold = 0.65  # Default, can be made configurable
    match_count = 5         # Default, can be made configurable
    max_retries = 3
    retry_delay_seconds = 2

    response_data = [] # Initialize outside loop

    for attempt in range(max_retries):
        try:
            print(f"Searching for similar products... (Attempt {attempt + 1} of {max_retries})")
            # The database call is now inside the loop, using your RPC function
            # Ensure your RPC function in Supabase is named 'similar_products'
            # and takes 'query_embedding', 'match_threshold', 'match_count'
            response = supabase_client.rpc(
                "similar_products",
                {
                    "query_embedding": query_embedding,
                    "match_threshold": match_threshold,
                    "match_count": match_count
                }
            ).execute()

            response_data = response.data # Get the actual data
            break # Success, break out of retry loop

        except Exception as e:
            print(f"Supabase RPC error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay_seconds} seconds...")
                time.sleep(retry_delay_seconds)
            else:
                # All retries failed
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail=f"Failed to fetch recommendations from Supabase after {max_retries} attempts. Service might be temporarily unavailable. Error: {e}"
                )

    if not response_data:
        return RecommendationResponse(
            message="No products found above the similarity threshold. Try adjusting the threshold.",
            recommendations=[]
        )

    # Format recommendations for response
    recommendations = []
    for product in response_data:
        try:
            # Ensure price is a string
            price_str = str(product.get('Price', 'N/A'))
            recommendations.append(ProductRecommendation(
                Product=product['Product'],
                Image=product['Image'],
                URL=product['URL'],
                Price=price_str,
                similarity=float(product['similarity'])
            ))
        except Exception as e:
            print(f"Error parsing product data: {product}. Error: {e}")
            # Optionally skip malformed products or log specific errors
            continue

    return RecommendationResponse(
        message=f"Successfully found {len(recommendations)} recommendations.",
        recommendations=recommendations
    )

