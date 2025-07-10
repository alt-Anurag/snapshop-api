# SnapShop Recommendation API

A FastAPI-based service for image-based product recommendations using CLIP embeddings and Supabase vector search.

---

## Features

- Accepts base64-encoded image input
- Generates CLIP embeddings (`clip-ViT-B-32`)
- Fetches similar products using Supabase RPC
- CORS-enabled and environment-variable configurable

---

## API Usage

### `POST /recommendations`

**Request:**
```json
{
  "image_data": "data:image/jpeg;base64,..."
}
```

**Response:**
```json
{
  "message": "Successfully found X recommendations.",
  "recommendations": [
    {
      "Product": "Example",
      "Image": "https://example.com/image.jpg",
      "URL": "https://example.com",
      "Price": "Rs. 999",
      "similarity": 0.82
    }
  ]
}
```

---

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
ALLOWED_ORIGINS=http://localhost,http://yourfrontend.com
```

3. Run the API:
```bash
uvicorn app:app --reload
```

---


