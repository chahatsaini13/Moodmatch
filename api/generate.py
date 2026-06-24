import os, re, json, time, base64
import concurrent.futures
from io import BytesIO
from http.server import BaseHTTPRequestHandler
import requests
from PIL import Image, ImageDraw, ImageFont
from groq import Groq


#  PATHS 
ROOT          = os.path.dirname(os.path.abspath(__file__))
DATA_PATH     = os.path.join(ROOT, "data",   "emotion_design_dataset.json")
TEMPLATE_PATH = os.path.join(ROOT, "assets", "Blank Template.png")
FONTS_DIR     = "/tmp/fonts"


#  API KEYS  
GROQ_API_KEY         = os.environ.get("GROQ_API_KEY")
HF_TOKEN             = os.environ.get("HF_TOKEN")
PEXELS_API_KEY       = os.environ.get("PEXELS_API_KEY")
GOOGLE_FONTS_API_KEY = os.environ.get("GOOGLE_FONTS_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)


#  ASSETS  
def _sanitize_label(label: str) -> str:
    return re.sub(r'[\s\-_\/]+', '/', label.strip().upper())

with open(DATA_PATH) as _f:
    _raw = json.load(_f)

emotion_dataset: dict = {_sanitize_label(k): v for k, v in _raw.items()}
template_bg: Image.Image = Image.open(TEMPLATE_PATH).convert("RGB")
EMOTION_MAP = {
    "JOY":       "JOY/HAPPINESS",
    "HAPPINESS": "JOY/HAPPINESS",
    "HAPPY":     "JOY/HAPPINESS",
    "FEAR":      "FEAR/ANXIETY",
    "ANXIETY":   "FEAR/ANXIETY",
    "DISGUST":   "DISGUST/NAUSEA",
    "NAUSEA":    "DISGUST/NAUSEA",
    "GUILT":     "GUILT/SHAME",
    "SHAME":     "GUILT/SHAME",
    "HOPE":      "HOPE/GLOW",
    "GLOW":      "HOPE/GLOW",
    "LOVE":      "LOVE/BLUSH",
    "BLUSH":     "LOVE/BLUSH",
    "NEUTRAL":   "NEUTRAL/CLARITY",
    "CLARITY":   "NEUTRAL/CLARITY",
    "PEACE":     "PEACE/SERENITY",
    "SERENITY":  "PEACE/SERENITY",
    "POWER":     "POWER/ANGER",
    "ANGER":     "POWER/ANGER",
    "ANGRY":     "POWER/ANGER",
    "SORROW":    "SORROW/DUSK",
    "SADNESS":   "SORROW/DUSK",
    "SAD":       "SORROW/DUSK",
    "SURPRISE":  "SURPRISE/WONDER",
    "WONDER":    "SURPRISE/WONDER",
    "TRUST":     "TRUST/SECURITY",
    "SECURITY":  "TRUST/SECURITY",
}

#  EMOTION CLASSIFICATION
def predict_emotion(text: str) -> str:
    import httpx
    url     = "https://api-inference.huggingface.co/models/chahatsaini1309/moodmatch-emotion-model"
    headers = {
        "Authorization":    f"Bearer {HF_TOKEN}",
        "x-wait-for-model": "true",
        "Content-Type":     "application/json",
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            client.post(url, headers=headers, json={"inputs": "warmup"})
    except Exception:
        pass

    backoff_schedule = [5, 15, 30, 45]

    for attempt, wait in enumerate(backoff_schedule):
        try:
            with httpx.Client(timeout=60.0) as client:
                resp = client.post(url, headers=headers, json={"inputs": text})
            if resp.status_code == 200:
                result = resp.json()
                if isinstance(result, list):
                    top = max(result[0], key=lambda x: x["score"])
                    label     = _sanitize_label(top["label"])
                    hf_emotion = EMOTION_MAP.get(label, label)
                    confidence = top["score"]

                    # HF is confident — trust it directly
                    if confidence >= 0.60:
                        return hf_emotion

                    # HF is unsure — let Groq decide
                    return _groq_fallback(text)

            if attempt < len(backoff_schedule) - 1:
                time.sleep(wait)
        except Exception:
            if attempt < len(backoff_schedule) - 1:
                time.sleep(wait)
            continue

    # HF completely down — Groq takes over
    return _groq_fallback(text)


def _groq_fallback(text: str) -> str:
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an emotion classifier. Given any word or phrase, return ONLY one of these exact labels — nothing else:\n"
                        "JOY/HAPPINESS, LOVE/BLUSH, SORROW/DUSK, POWER/ANGER, FEAR/ANXIETY, "
                        "PEACE/SERENITY, TRUST/SECURITY, HOPE/GLOW, SURPRISE/WONDER, "
                        "GUILT/SHAME, DISGUST/NAUSEA, NEUTRAL/CLARITY"
                    )
                },
                {"role": "user", "content": text}
            ],
        )
        label = resp.choices[0].message.content.strip().upper()
        if label in emotion_dataset:
            return label
    except Exception:
        pass
    return "NEUTRAL/CLARITY"


#  DESIGN EXTRACTION
def extract_design(emotion: str):
    design  = emotion_dataset[emotion]
    colors  = design["colors"]
    palette = list(colors.values()) if isinstance(colors, dict) else colors
    fonts   = design.get("fonts", {})
    return palette, fonts.get("title", "Playfair Display"), fonts.get("body", "Lato")


#  FONT DOWNLOAD
_FONT_URL_OVERRIDES = {
    "Poppins":    "https://fonts.gstatic.com/s/poppins/v20/pxiGyp8kv8JHgFVrJJfedw.woff2",
    "Open Sans":  "https://fonts.gstatic.com/s/opensans/v18/mem8YaGs126MiZpBA-UFVZ0bf8pkAg.woff2",
}

def _sanitize_font_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum()).lower()

def _fetch_font_url(font_name: str) -> str | None:
    if font_name in _FONT_URL_OVERRIDES:
        return _FONT_URL_OVERRIDES[font_name]
    try:
        resp  = requests.get(
            "https://www.googleapis.com/webfonts/v1/webfonts",
            params={"key": GOOGLE_FONTS_API_KEY},
            timeout=10,
        )
        items = resp.json().get("items", [])
        for item in items:
            if item["family"].lower() == font_name.lower():
                files = item.get("files", {})
                return files.get("regular") or next(iter(files.values()), None)
    except Exception:
        pass
    return None

def _download_font(name: str, url: str) -> str:
    slug = _sanitize_font_filename(name)
    path = os.path.join(FONTS_DIR, f"{slug}.ttf")
    if os.path.exists(path):
        return path
    os.makedirs(FONTS_DIR, exist_ok=True)
    data = requests.get(url, timeout=15).content
    with open(path, "wb") as fh:
        fh.write(data)
    return path

def load_fonts(emotion: str) -> dict:
    data      = emotion_dataset[emotion]
    font_meta = data.get("fonts", {})
    paths     = {}
    for role in ("headings", "body_text", "highlight_text"):
        raw_name  = font_meta.get(role, "Open Sans").split(" (")[0].strip()
        font_url  = _fetch_font_url(raw_name)
        if font_url:
            paths[role] = _download_font(raw_name, font_url)
    return paths


#  CANVAS HELPERS
def create_canvas() -> Image.Image:
    return template_bg.copy()

def fit_image(img: Image.Image, target_size: tuple) -> Image.Image:
    img = img.convert("RGB")
    tw, th = target_size
    ir = img.width / img.height
    tr = tw / th
    if ir > tr:
        nw   = int(img.height * tr)
        left = (img.width - nw) // 2
        img  = img.crop((left, 0, left + nw, img.height))
    else:
        nh  = int(img.width / tr)
        top = (img.height - nh) // 2
        img = img.crop((0, top, img.width, top + nh))
    return img.resize(target_size, Image.LANCZOS)

def render_colors(canvas: Image.Image, palette: list) -> Image.Image:
    draw   = ImageDraw.Draw(canvas)
    blocks = [
        (360, 1120, 488, 1248),
        (508, 1120, 636, 1248),
        (656, 1120, 784, 1248),
        (804, 1120, 932, 1248),
        (952, 1120, 1080, 1248),
    ]
    for i, box in enumerate(blocks):
        draw.rectangle(box, fill=palette[i % len(palette)])
    return canvas

def _draw_wrapped_label(draw, text, x1, x2, y, font, fill):
    cx = (x1 + x2) // 2
    if "(" in text:
        main = text.split("(")[0].strip()
        sub  = "(" + text.split("(", 1)[1]
        for part, dy in ((main, 0), (sub, 28)):
            bb = draw.textbbox((0, 0), part, font=font)
            w  = bb[2] - bb[0]
            draw.text((cx - w // 2, y + dy), part, fill=fill, font=font)
        return y + 50
    bb = draw.textbbox((0, 0), text, font=font)
    w  = bb[2] - bb[0]
    draw.text((cx - w // 2, y), text, fill=fill, font=font)
    return y + 28

def render_typography(canvas: Image.Image, emotion: str) -> Image.Image:
    font_paths   = load_fonts(emotion)
    draw         = ImageDraw.Draw(canvas)
    data         = emotion_dataset[emotion]
    bg_color     = data.get("font_bg_color",  "#DDDDDD")
    font_color   = data.get("font_color",      "#FFFFFF")
    x1, y1, x2, y2 = 0, 520, 530, 1095
    draw.rectangle((x1, y1, x2, y2), fill=bg_color)

    def _tfont(role, size):
        path = font_paths.get(role)
        return ImageFont.truetype(path, size) if path else ImageFont.load_default()

    heading_font   = _tfont("headings",       65)
    body_font      = _tfont("body_text",       55)
    highlight_font = _tfont("highlight_text",  65)
    label_font     = _tfont("body_text",       25)

    emotion_word = emotion.split("/")[0].title()
    sample_text  = f"Feel The {emotion_word}"
    fonts_meta   = data.get("fonts", {})
    y = 580

    for role, fnt, offset in (
        ("headings",       heading_font,   0),
        ("body_text",      body_font,      170),
        ("highlight_text", highlight_font, 320),
    ):
        label = fonts_meta.get(role, role.replace("_", " ").title())
        ny    = _draw_wrapped_label(draw, label, x1, x2, y + offset, label_font, font_color)
        bb    = draw.textbbox((0, 0), sample_text, font=fnt)
        w     = bb[2] - bb[0]
        draw.text(((x1 + x2) // 2 - w // 2, ny + 25), sample_text, fill=font_color, font=fnt)

    return canvas


#  IMAGE SEARCH & FETCH
def build_image_queries(prompt: str, emotion: str) -> list[str]:
    palette, title_font, body_font = extract_design(emotion)
    context = f"""
    You are a visual search expert. Generate 6 Pexels photography search queries for a moodboard.
    
    User's idea: {prompt}
    Emotional tone: {emotion}
    Color palette to match: {palette}
    
    RULES:
    - First, extract the concrete subject, industry, or domain from the user's idea — the literal thing it's about, not the feeling behind it.
    - At least 4 of the 6 queries MUST contain a literal, visual object, person, or scene that belongs to that domain. If someone outside the project couldn't tell what industry or topic the moodboard is for just by looking at the images, the queries have failed.
    - The emotional tone ({emotion}) should shape HOW that domain subject is shot — lighting, energy, atmosphere — not REPLACE it with an unrelated abstract scene.
    - The remaining 1-2 queries may be more atmospheric or supporting shots, but should still relate to the domain where possible.
    - Each query must incorporate the colors from the palette: {palette} — through lighting, objects, clothing, backgrounds, or atmosphere
    - The brightness, energy, and mood of every query must match the emotion — bright and airy for positive emotions, dark and muted for negative ones
    - Use concrete, visual, real-world subjects: people, nature, objects, spaces, textures
    - Add photography style words: natural light, cinematic, soft focus, golden hour, high contrast, shallow depth of field
    - Keep each query between 5 and 10 words
    - Do NOT use abstract or conceptual words like "emotion", "feeling", "mood", "concept"
    - Do NOT repeat the same subject across queries — vary the scenes
    
    Return ONLY valid JSON with key "queries" containing exactly 6 strings.
    """
    resp = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a visual search expert who creates photography search queries for moodboards. "
                    "Queries must visually and emotionally match the given tone and color palette. "
                    "The lighting, energy, and subject matter must reflect the emotion accurately. "
                    "Return JSON with key 'queries' containing exactly 6 strings."
                )
            },
            {"role": "user", "content": context},
        ],
    )
    return json.loads(resp.choices[0].message.content)["queries"]
    

def _fetch_one(query: str) -> Image.Image | None:
    try:
        r    = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": PEXELS_API_KEY},
            params={"query": query, "per_page": 1},
            timeout=15,
        )
        data = r.json()
        if not data.get("photos"):
            return None
        url      = data["photos"][0]["src"]["large"]
        img_data = requests.get(url, timeout=15).content
        return Image.open(BytesIO(img_data))
    except Exception:
        return None

def fetch_images_parallel(queries: list[str]) -> list[Image.Image]:
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_fetch_one, queries[:6]))
    return [img for img in results if img is not None]

def render_images(canvas: Image.Image, images: list) -> Image.Image:
    slots = [
        (0,   0,    533, 496),
        (551, 0,    1080, 469),
        (551, 496,  1080, 1097),
        (0,   1120, 341,  1620),
        (361, 1271, 784,  1620),
        (804, 1271, 1080, 1620),
    ]
    for i, img in enumerate(images[:6]):
        x1, y1, x2, y2 = slots[i]
        canvas.paste(fit_image(img, (x2 - x1, y2 - y1)), (x1, y1))
    return canvas


#  MAIN PIPELINE
def create_moodboard(prompt: str, emotion: str) -> Image.Image:
    palette, _, _ = extract_design(emotion)
    canvas        = create_canvas()
    canvas        = render_colors(canvas, palette)
    canvas        = render_typography(canvas, emotion)
    queries       = build_image_queries(prompt, emotion)
    images        = fetch_images_parallel(queries)
    canvas        = render_images(canvas, images)
    return canvas


#  VERCEL HTTP HANDLER
class handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            body   = json.loads(self.rfile.read(length))
            prompt = body.get("prompt", "").strip()

            if not prompt:
                return self._json(400, {"error": "prompt is required"})

            emotion = predict_emotion(prompt)
            board   = create_moodboard(prompt, emotion)

            buf = BytesIO()
            board.save(buf, format="PNG", optimize=True)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            self._json(200, {"image_b64": img_b64, "emotion": emotion})

        except Exception as e:
            self._json(500, {"error": str(e)})

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin",  "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, code: int, data: dict):
        payload = json.dumps(data).encode()
        self.send_response(code)
        self._cors_headers()
        self.send_header("Content-Type",   "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
