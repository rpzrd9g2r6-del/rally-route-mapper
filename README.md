# Rally Route Mapper

Browser-based prototype for converting rally route-map PDFs/images into route data.

## Local run
pip install -r requirements.txt
uvicorn app.main:app --reload

Then open http://127.0.0.1:8000

## Render
Create a new Web Service from this repository. Render will use render.yaml.
Set GOOGLE_MAPS_API_KEY as a secret environment variable.

## Important
The Google Maps API key must be restricted in Google Cloud and must never be committed to this repository.
