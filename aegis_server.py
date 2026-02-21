from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse
import httpx  # For forwarding the request to the real LLM if cache misses
import time
from typing import Optional

# Import the mathematical engine we just built
from aegis_core_math import AegisAlgorithmicEngine

app = FastAPI(title="AegisCache Enterprise Gateway")

# Initialize our proprietary mathematical cache
cache_engine = AegisAlgorithmicEngine(alpha=0.7, beta=0.3)

# The real LLM destination
REAL_LLM_URL = "https://api.openai.com/v1/chat/completions"

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_tenant_id: Optional[str] = Header(None, alias="X-Tenant-ID"),
    x_role_id: Optional[str] = Header(None, alias="X-Role-ID")
):
    """
    This endpoint perfectly mimics the OpenAI API.
    LangChain/LangGraph will send requests here without realizing it's a proxy.
    """
    # 1. Enforce RBAC Security at the Gateway
    if not x_tenant_id or not x_role_id:
        raise HTTPException(status_code=401, detail="Enterprise Gateway Error: Missing RBAC Headers (X-Tenant-ID, X-Role-ID)")

    body = await request.json()
    
    # 2. Extract the actual user prompt (last message in the array)
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="No messages found.")
    
    user_prompt = messages[-1].get("content", "")

    start_time = time.time()

    # ==========================================
    # 3. INTERCEPT: Check the Proprietary Cache
    # ==========================================
    cached_answer, score = cache_engine.search_cache(
        tenant_id=x_tenant_id, 
        role_id=x_role_id, 
        prompt=user_prompt
    )

    if cached_answer:
        # CACHE HIT! Return instantly mimicking OpenAI's JSON structure
        latency = (time.time() - start_time) * 1000
        print(f"⚡ CACHE HIT! Score: {score:.2f} | Latency: {latency:.2f}ms")
        
        return JSONResponse(content={
            "id": "chatcmpl-aegiscache-hit",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "aegis-cache-engine",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": cached_answer
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "aegis_saved_money": True # A fun custom flag for your POC
            }
        })

    # ==========================================
    # 4. CACHE MISS: Forward to the Real LLM
    # ==========================================
    print(f"🐌 CACHE MISS. Score: {score:.2f}. Forwarding to actual LLM...")
    
    # Extract the original authorization header (The OpenAI API Key)
    auth_header = request.headers.get("Authorization")
    
    async with httpx.AsyncClient() as client:
        llm_response = await client.post(
            REAL_LLM_URL,
            headers={"Authorization": auth_header, "Content-Type": "application/json"},
            json=body,
            timeout=60.0
        )
        
    if llm_response.status_code != 200:
        return JSONResponse(status_code=llm_response.status_code, content=llm_response.json())
        
    llm_data = llm_response.json()
    actual_answer = llm_data["choices"][0]["message"]["content"]

    # ==========================================
    # 5. ASYNC WRITE: Save the new answer to Cache
    # ==========================================
    # In a fully production app, we would fire this off as a BackgroundTask so 
    # the user doesn't wait for the write operation.
    cache_engine.insert_to_cache(
        tenant_id=x_tenant_id,
        role_id=x_role_id,
        prompt=user_prompt,
        response=actual_answer
    )

    return JSONResponse(content=llm_data)

if __name__ == "__main__":
    import uvicorn
    # Runs the proxy server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
    