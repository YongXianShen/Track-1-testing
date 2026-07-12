"""Very small Fireworks batch prompts for V17.6."""
from __future__ import annotations
import json, re
from typing import Any
from . import prompts

CODES = {"factual":"F","math":"M","sentiment":"S","summary":"U","ner":"N","logic":"L","debug":"D","codegen":"C"}
SYS_GENERAL = "JSON {id:answer}. Solve independently. Complete, exact requested format, concise. M include final units; S label+reason; N all entities; L all constraints."
SYS_CODE = "JSON {id:answer}. D: bug+minimal fix. C: minimal correct code, edge cases, no fences."

def _source_tail(prompt: str) -> str:
    quoted = re.findall(r"['\"]([^'\"]{10,})['\"]", prompt, re.S)
    return quoted[-1].strip() if quoted else prompt.strip()

def compact(item: dict[str,str]) -> dict[str,Any]:
    p=item["prompt"]; c=item["category"]
    if c in {"sentiment","ner"}: return {"i":item["task_id"],"c":CODES[c],"q":_source_tail(p)}
    if c=="summary":
        rule = " ".join(re.findall(r"exactly\s+\w+\s+(?:sentences?|bullet points?|bullets?|points?|words?)|(?:each|per).{0,25}(?:at most|under|no longer than)\s+\d+\s+words?", p, re.I))
        return {"i":item["task_id"],"c":"U","r":rule,"q":_source_tail(p)}
    return {"i":item["task_id"],"c":CODES[c],"q":p.strip()}

def messages(items:list[dict[str,str]], code:bool)->list[dict[str,str]]:
    data=[compact(x) for x in items]
    return [{"role":"system","content":SYS_CODE if code else SYS_GENERAL},{"role":"user","content":json.dumps(data,ensure_ascii=False,separators=(",",":"))}]

def cap(items:list[dict[str,str]], code:bool)->int:
    per={"factual":55,"math":45,"sentiment":30,"summary":80,"ner":60,"logic":55,"debug":120,"codegen":150}
    return min(320 if code else 220, 12+sum(per[x["category"]] for x in items))

def parse(text:str, expected:set[str], categories:dict[str,str])->dict[str,str]:
    text=re.sub(r"<think>.*?</think>","",text or "",flags=re.I|re.S).strip()
    text=re.sub(r"^```(?:json)?\s*|\s*```$","",text,flags=re.I)
    obj=None
    try: obj=json.loads(text)
    except Exception:
        a,b=text.find("{"),text.rfind("}")
        if a>=0 and b>a:
            try: obj=json.loads(text[a:b+1])
            except Exception: pass
    raw={}
    if isinstance(obj,dict): raw=obj
    elif isinstance(obj,list):
        for row in obj:
            if isinstance(row,dict) and ("id" in row or "i" in row) and ("answer" in row or "a" in row): raw[str(row.get("id",row.get("i")))]=row.get("answer",row.get("a"))
    out={}
    for i in expected:
        v=raw.get(i)
        if isinstance(v,(str,int,float)) and str(v).strip(): out[i]=prompts.postprocess(categories[i],str(v)).strip()
    return out
