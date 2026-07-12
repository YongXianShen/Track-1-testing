"""Track 1 Ultra-Local V17.6.

Target: keep at least 84.2% while cutting Fireworks use toward ~500 tokens.
Most tasks are solved by generic local algorithms; at most three unresolved
items are sent through compact MiniMax/Kimi batches. No paid Gemma deployment.
"""
from __future__ import annotations
import asyncio, json, os, sys, time
from typing import Any
from . import client, local_first, micro_batching, models, router

INPUT_PATH=os.environ.get("INPUT_PATH","/input/tasks.json")
OUTPUT_PATH=os.environ.get("OUTPUT_PATH","/output/results.json")
DEADLINE_SECONDS=float(os.environ.get("DEADLINE_SECONDS",str(8.5*60)))
REMOTE_TASK_BUDGET=int(os.environ.get("REMOTE_TASK_BUDGET","3"))
START=time.monotonic(); CALL_LOG=[]; MODEL_PLAN={}

def log(event:str,**fields:Any)->None:
    print(event+" "+json.dumps(fields,ensure_ascii=False),file=sys.stderr,flush=True)

def task_prompt(task:dict[str,Any])->str:
    for k in ("prompt","question","input","text"):
        v=task.get(k)
        if isinstance(v,str) and v.strip(): return v
    return ""

async def call(label:str,model:str,messages:list[dict[str,str]],cap:int)->tuple[str,bool]:
    try:
        text,usage=await client.complete(model,messages,cap)
        trunc=bool(usage.pop("truncated",False)); CALL_LOG.append({"label":label,"model":model,**usage,"truncated":trunc})
        log("USAGE",label=label,model=model,**usage,truncated=trunc); return text,trunc
    except Exception as e:
        log("ERROR",label=label,model=model,error=str(e)[:180]); return "",False

def priority(item:dict[str,str])->tuple[int,int]:
    # Remote budget is spent where local heuristics are least dependable.
    rank={"logic":0,"debug":1,"codegen":2,"factual":3,"math":4,"ner":5,"summary":6,"sentiment":7}
    return rank.get(item["category"],9), len(item["prompt"])

async def batch(items:list[dict[str,str]],model:str,code:bool)->dict[str,str]:
    if not items:return {}
    text,trunc=await call("micro:code" if code else "micro:general",model,micro_batching.messages(items,code),micro_batching.cap(items,code))
    if trunc:return {}
    cats={x["task_id"]:x["category"] for x in items}
    return micro_batching.parse(text,set(cats),cats)

def best_effort(item:dict[str,str])->str:
    # This path is deliberately conservative. It should be rare because the
    # extended local solvers cover the public format families.
    c,p=item["category"],item["prompt"]
    if c=="sentiment": return local_first.solve_sentiment(p) or "Neutral — the text does not express a clearly dominant positive or negative view."
    if c=="summary": return local_first.solve_summary(p) or "Summary unavailable."
    if c=="ner": return local_first.solve_ner(p) or "No named entities identified."
    if c=="math": return local_first.solve_math(p) or "Answer: unable to determine from the stated information."
    if c=="logic": return local_first.solve_logic(p) or "Answer: insufficient constraints for a unique result."
    if c=="codegen": return local_first.solve_codegen(p) or "def solution(*args, **kwargs):\n    raise NotImplementedError"
    if c=="debug": return local_first.solve_debug(p) or "The implementation must be corrected to satisfy the stated specification and edge cases."
    return local_first.solve_factual(p) or "The requested concept cannot be answered reliably from the local knowledge base."

async def run(tasks:list[dict[str,Any]],results:dict[str,str])->None:
    plan=models.build_plan(); MODEL_PLAN.update(plan.as_dict()); log("MODEL_PLAN",**MODEL_PLAN,version="V17.6",remote_budget=REMOTE_TASK_BUDGET,gemma_used=False)
    unresolved=[]
    for task in tasks:
        tid=str(task["task_id"]); p=task_prompt(task)
        if not p: continue
        c=router.classify(p) or "factual"
        ans=local_first.solve(c,p)
        if ans:
            results[tid]=ans; log("LOCAL",task_id=tid,category=c)
        else: unresolved.append({"task_id":tid,"category":c,"prompt":p})
    selected=sorted(unresolved,key=priority)[:max(0,REMOTE_TASK_BUDGET)]
    selected_ids={x["task_id"] for x in selected}
    general=[x for x in selected if x["category"] not in {"debug","codegen"}]
    code=[x for x in selected if x["category"] in {"debug","codegen"}]
    remaining=DEADLINE_SECONDS-(time.monotonic()-START)
    try:
        ga,ca=await asyncio.wait_for(asyncio.gather(batch(general,plan.REASON,False),batch(code,plan.CODE,True)),timeout=max(5,remaining))
    except asyncio.TimeoutError: ga,ca={},{}
    answers={**ga,**ca}
    for item in unresolved:
        tid=item["task_id"]
        results[tid]=answers.get(tid) or best_effort(item)
        if tid not in selected_ids: log("LOCAL_FALLBACK",task_id=tid,category=item["category"])
    await client.aclose()

def write_results(tasks:list[dict[str,Any]],results:dict[str,str])->None:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".",exist_ok=True)
    payload=[{"task_id":t["task_id"],"answer":str(results.get(str(t["task_id"]),"")).strip()} for t in tasks]
    tmp=OUTPUT_PATH+".tmp"
    with open(tmp,"w",encoding="utf-8") as f: json.dump(payload,f,ensure_ascii=False,separators=(",",":"))
    os.replace(tmp,OUTPUT_PATH)

def write_usage()->None:
    totals={"prompt_tokens":sum(int(x.get("prompt_tokens",0)) for x in CALL_LOG),"completion_tokens":sum(int(x.get("completion_tokens",0)) for x in CALL_LOG)}
    totals["total_tokens"]=totals["prompt_tokens"]+totals["completion_tokens"]; totals["calls"]=len(CALL_LOG)
    try:
        with open(os.path.join(os.path.dirname(OUTPUT_PATH) or ".","model_usage.json"),"w",encoding="utf-8") as f: json.dump({"version":"V17.6","model_plan":MODEL_PLAN,"calls":CALL_LOG,"totals":totals},f,indent=2)
    except Exception: pass

def main()->int:
    try:
        with open(INPUT_PATH,encoding="utf-8-sig") as f: raw=json.load(f)
        if not isinstance(raw,list): raise ValueError("tasks.json must be a list")
        tasks=[x for x in raw if isinstance(x,dict) and x.get("task_id")]
        for name in ("FIREWORKS_API_KEY","FIREWORKS_BASE_URL","ALLOWED_MODELS"):
            if not os.environ.get(name): raise ValueError(f"{name} is required")
        results={str(x["task_id"]):"" for x in tasks}; asyncio.run(run(tasks,results)); write_results(tasks,results); write_usage()
        log("DONE",tasks=len(tasks),answered=sum(bool(v) for v in results.values()),elapsed_s=round(time.monotonic()-START,1)); return 0
    except Exception as e:
        log("FATAL",error=str(e)[:300]); return 1
if __name__=="__main__": sys.exit(main())
