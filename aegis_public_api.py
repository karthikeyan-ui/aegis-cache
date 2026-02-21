from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import JSONResponse
import time
from aegis_saas_engine import AegisSaaSEngine

app = FastAPI(title="AegisCache Public SaaS API")

# Boot up the robust, persistent engine
saas_engine = AegisSaaSEngine()

# --- MOCK DATABASE FOR SUBSCRIPTIONS ---
# In reality, this connects to PostgreSQL and Stripe
VALID_API_KEYS = {
    "aegis_live_demo_123": {"company": "Startup_A", "tier": "pro", "requests_used": 0, "limit": 10000},
    "aegis_live_demo_999": {"company": "Enterprise_B", "tier": "enterprise", "requests_used": 0, "limit": 500000}
}

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_subscription(api_key: str = Security(api_key_header)):
    """Middleware to validate payment and API keys."""
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API Key. Sign up at aegiscache.com")
    
    key_clean = api_key.replace("Bearer ", "")
    
    client_data = VALID_API_KEYS.get(key_clean)
    if not client_data:
        raise HTTPException(status_code=403, detail="Invalid API Key.")
        
    if client_data["requests_used"] >= client_data["limit"]:
        raise HTTPException(status_code=402, detail="Payment Required. Request limit reached.")
        
    return client_data, key_clean

@app.post("/v1/chat/completions")
async def public_chat_completions(
    request: Request, 
    client_info: tuple = Depends(verify_subscription)
):
    client_data, api_key = client_info
    company_tenant_id = client_data["company"] # We use their company name as the security tenant
    
    # 1. Billing: Increment their usage map (to charge them later via Stripe)
    VALID_API_KEYS[api_key]["requests_used"] += 1

    body = await request.json()
    messages = body.get("messages", [])
    user_prompt = messages[-1].get("content", "")

    # Note: For the public API, we extract the "Role" from an optional custom header. 
    # If they don't provide it, we default to a "global" role for that company.
    user_role = request.headers.get("X-User-Role", "default_global_role")

    start_time = time.time()

    # 2. Check the Proprietary SaaS Cache
    cached_answer, score = saas_engine.search_cache(
        tenant_id=company_tenant_id, 
        role_id=user_role, 
        prompt=user_prompt
    )

    if cached_answer:
        # RETURN INSTANTLY -> You just saved them OpenAI costs!
        return JSONResponse(content={
            "id": "chatcmpl-aegis-hit",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "aegis-cache-pro",
            "choices": [{"message": {"role": "assistant", "content": cached_answer}, "finish_reason": "stop"}],
            "aegis_metadata": {
                "cache_status": "HIT",
                "latency_ms": round((time.time() - start_time) * 1000, 2),
                "credits_remaining": client_data["limit"] - VALID_API_KEYS[api_key]["requests_used"]
            }
        })

    # ==========================================
    # CACHE MISS LOGIC
    # ==========================================
    # In a true SaaS Gateway, you would forward the request to OpenAI here using their key,
    # get the response, write it to your saas_engine, and return it.
    # (Leaving this out for brevity, but it's the exact same httpx logic from our earlier server).
    
    return JSONResponse(content={"error": "Cache miss. LLM Forwarding goes here."})

@app.post("/v1/admin/snapshot")
async def force_backup():
    """Admin endpoint to force save the math arrays to disk."""
    saas_engine.save_snapshot()
    return {"status": "success", "message": "Algorithm state persisted to disk."}
    