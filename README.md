# MoodMatch

MoodMatch is a web app that generates emotion-inspired moodboards from a text prompt. You describe a feeling or idea, and the app classifies the emotion, selects a matching color palette and typography, fetches relevant photography, and renders a downloadable moodboard image.

---

## How it works

1. User enters a text prompt on the generator page
2. A fine-tuned transformer model classifies the emotion behind the text
3. If the model is uncertain, Groq LLM verifies or corrects the prediction
4. A color palette, font set, and design style are pulled from a curated emotion-design dataset
5. Groq generates 6 photography search queries tailored to the emotion and palette
6. Images are fetched in parallel from Pexels
7. Pillow renders everything into a final moodboard image
8. The image is returned to the browser as a base64 PNG

---

## Tech stack

- Frontend: HTML, CSS, JavaScript
- Backend: Python serverless functions via Vercel
- Emotion classification: Fine-tuned transformer model hosted on HuggingFace Inference API
- LLM: Groq API (llama-3.3-70b-versatile) for emotion verification and query generation
- Images: Pexels API
- Fonts: Google Fonts API
- Rendering: Pillow (PIL)

---

## Environment variables

These must be set in your Vercel project dashboard before deploying.

| Variable | Description |
|---|---|
| `HF_TOKEN` | HuggingFace API token |
| `GROQ_API_KEY` | Groq API key |
| `PEXELS_API_KEY` | Pexels API key |
| `GOOGLE_FONTS_API_KEY` | Google Fonts API key |

---

## Emotion labels

The model classifies input into one of 12 emotion categories:

JOY/HAPPINESS, LOVE/BLUSH, SORROW/DUSK, POWER/ANGER, FEAR/ANXIETY, PEACE/SERENITY, TRUST/SECURITY, HOPE/GLOW, SURPRISE/WONDER, GUILT/SHAME, DISGUST/NAUSEA, NEUTRAL/CLARITY

Each emotion maps to a unique design profile in the dataset with its own color palette, font choices, and photography style.

---

## Deployment

The app is deployed on Vercel. The `vercel.json` file configures two Python serverless functions and static file serving. The generate function has a 120 second timeout to accommodate HuggingFace cold starts.

---

## Team

Chahat Saini, Navneet Sah, Yashika 
