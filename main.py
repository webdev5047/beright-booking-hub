import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="BEright AI Care Agent & Smart Appointment Hub")

APPS_SCRIPT_URL = "https://script.google.com/macros/s/AKfycbzFIofgR2dAcNQa-Sq3JQy7VULH-85iQspmb7j69HUWtKqdNeEFsRWNeN1CWiqLH2cK/exec"

# Runtime or .env AI Agent Keys (Free Gemini or Groq API)
AI_CONFIG = {
    "provider": os.getenv("AI_PROVIDER", "gemini").strip().lower(),
    "gemini_key": os.getenv("GEMINI_API_KEY", "").strip(),
    "groq_key": os.getenv("GROQ_API_KEY", "").strip(),
}

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


class BookingPayload(BaseModel):
    token: str
    mode: str
    name: str
    phone: str
    email: Optional[str] = ""
    service: str
    price: float
    currency: str = "₹"
    date: str
    slot: str


class SlotPayload(BaseModel):
    slotTime: str


class StatusPayload(BaseModel):
    token: str
    status: str


class ApiKeyConfigPayload(BaseModel):
    provider: str
    apiKey: str


class WeeklyCampaignPayload(BaseModel):
    mode: str = "clinic"
    useAiDraft: bool = True


class ChatQuery(BaseModel):
    question: str
    mode: str = "clinic"
    service: str = ""
    date: str = ""
    freeSlots: list[str] = []


async def forward_to_apps_script(payload: dict) -> dict:
    async with httpx.AsyncClient(follow_redirects=False, timeout=25.0) as client:
        res = await client.post(APPS_SCRIPT_URL, json=payload)
        if res.status_code in (301, 302):
            redirect_url = res.headers.get("Location")
            if redirect_url:
                final_res = await client.get(redirect_url)
                return final_res.json()
        return res.json()


async def run_llm_agent(system_prompt: str, user_prompt: str) -> Optional[str]:
    """Calls Free Gemini or Groq API if configured; returns None if no key is set yet."""
    provider = AI_CONFIG["provider"]
    gemini_key = AI_CONFIG["gemini_key"]
    groq_key = AI_CONFIG["groq_key"]

    async with httpx.AsyncClient(timeout=18.0) as client:
        # 1. Try Groq (Llama 3.3 70B Free Tier)
        if (provider == "groq" and groq_key) or (not gemini_key and groq_key):
            try:
                res = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.4,
                        "max_tokens": 260,
                    },
                )
                if res.status_code == 200:
                    return res.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                pass

        # 2. Try Google Gemini (Gemini 2.5 Flash Free Tier)
        if gemini_key:
            try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={gemini_key}"
                res = await client.post(
                    url,
                    json={
                        "contents": [{"parts": [{"text": f"{system_prompt}\n\nUser: {user_prompt}"}]}],
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    return data["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                pass

    return None


@app.get("/")
async def home():
    return FileResponse("static/index.html")


@app.get("/api/state")
async def get_system_state():
    ai_active = bool(AI_CONFIG["gemini_key"] or AI_CONFIG["groq_key"])
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0) as client:
            res = await client.get(APPS_SCRIPT_URL)
            data = res.json()
            data["aiAgentActive"] = ai_active
            data["aiProvider"] = AI_CONFIG["provider"]
            return data
    except Exception:
        return {"bookings": [], "clients": [], "brain": [], "customSlots": [], "aiAgentActive": ai_active}


@app.post("/api/config-ai")
async def configure_ai_key(cfg: ApiKeyConfigPayload):
    prov = cfg.provider.strip().lower()
    key = cfg.apiKey.strip()
    if prov == "groq":
        AI_CONFIG["provider"] = "groq"
        AI_CONFIG["groq_key"] = key
    else:
        AI_CONFIG["provider"] = "gemini"
        AI_CONFIG["gemini_key"] = key
    return {"status": "saved", "provider": AI_CONFIG["provider"], "aiAgentActive": bool(key)}


@app.post("/api/book")
async def create_booking(data: BookingPayload):
    payload = data.model_dump()
    payload["action"] = "create_booking"
    return await forward_to_apps_script(payload)


@app.post("/api/slots")
async def add_custom_slot(data: SlotPayload):
    if not data.slotTime.strip():
        raise HTTPException(status_code=400, detail="Slot time required")
    return await forward_to_apps_script({"action": "add_slot", "slotTime": data.slotTime.strip()})


@app.post("/api/status")
async def update_booking_status(data: StatusPayload):
    return await forward_to_apps_script({"action": "update_status", "token": data.token, "status": data.status})


@app.post("/api/weekly-campaign")
async def trigger_weekly_care_emails(req: WeeklyCampaignPayload):
    custom_note = ""
    if req.useAiDraft:
        sys_p = "You are a warm, caring healthcare and wellness AI Care Agent for BEright Care Clinic & Salon."
        usr_p = (
            f"Write a short, friendly 3-sentence weekly check-in greeting for a {req.mode} client asking how they are feeling "
            f"after their recent visit, whether they have any ongoing concerns, and inviting them to check this week's doctor/specialist schedule below. Do not include a subject line."
        )
        ai_note = await run_llm_agent(sys_p, usr_p)
        if ai_note:
            custom_note = ai_note

    return await forward_to_apps_script({"action": "send_weekly_emails", "customNote": custom_note})


@app.post("/api/chat")
async def ai_care_agent_chat(q: ChatQuery):
    state = await get_system_state()
    brain_rows = state.get("brain", [])
    free_list = ", ".join(q.freeSlots[:6]) if q.freeSlots else "Multiple slots open today"

    # Build context from stored Q&A knowledge + live slots
    kb_context = "\n".join([f"Q: {r.get('Question')} | A: {r.get('Answer')}" for r in brain_rows[-15:]])

    system_prompt = (
        f"You are BEright Proactive AI Care & Booking Agent for a {q.mode} center.\n"
        f"Current selected service/doctor: {q.service}. Date: {q.date}. Open slots: {free_list}.\n"
        f"Weekly Doctor/Service Schedule:\n"
        f"- Dr. A. Sen (General Physician): Daily 09:00 AM - 09:00 PM\n"
        f"- Dr. R. Das (Dental): Tue, Thu, Sat, Sun\n"
        f"- Dr. P. Roy (Pediatrician): Mon, Wed, Fri, Sat\n"
        f"- Beauty & Skin Specialists: Daily 08:00 AM - 10:30 PM\n"
        f"Knowledge Base:\n{kb_context}\n\n"
        f"Instructions: Respond warmly in the user's language (English, Bengali, or Polish). "
        f"Proactively ask how the patient/client is feeling today, answer their question accurately using the live slots & schedule, "
        f"and guide them to click an open green slot to confirm their QR token."
    )

    llm_reply = await run_llm_agent(system_prompt, q.question)
    if llm_reply:
        return {"answer": llm_reply, "mode": "Autonomous AI Agent"}

    # Smart fallback when no API key is attached yet
    user_q = q.question.lower()
    best_answer = None
    best_overlap = 0
    user_words = set(w for w in user_q.split() if len(w) > 2)

    for row in brain_rows:
        bq = str(row.get("Question", "")).lower()
        ba = str(row.get("Answer", ""))
        b_words = set(w for w in bq.split() if len(w) > 2)
        overlap = len(user_words.intersection(b_words))
        if overlap > best_overlap and overlap >= 2:
            best_overlap = overlap
            best_answer = ba

    if best_answer:
        return {"answer": f"{best_answer} How are you feeling today—would you like me to help you pick a time slot?"}

    return {
        "answer": f"Hello! How are you feeling today? For {q.service} on {q.date}, open slots include: {free_list}. Our doctors & specialists are available all week—click any green slot to get your instant QR Token!"
    }