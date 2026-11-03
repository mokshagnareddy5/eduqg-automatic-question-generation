"""Automatic Question Generation from educational text.

Runs with the Python standard library. Optional transformer generation uses
Hugging Face Transformers and a compatible PyTorch installation.
"""
from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Iterable
from urllib.parse import parse_qs, urlparse


@dataclass
class Question:
    question: str
    answer: str
    evidence: str
    method: str
    relevance: int
    grammaticality: int
    answerability: int
    explanation: str


SAMPLES = {
    "biology": {
        "title": "Biology: Photosynthesis",
        "text": ("Photosynthesis is the process by which green plants convert light energy into chemical energy. "
                 "Chlorophyll is a pigment found in chloroplasts. A chloroplast contains chlorophyll and other structures. "
                 "Photosynthesis occurs mainly in the leaves of a plant. Sunlight provides the energy needed for the reaction."),
    },
    "history": {
        "title": "History: Ancient Rome",
        "text": ("Roman history is the study of ancient Rome and its political institutions. "
                 "The Roman Empire is an ancient state centered on the city of Rome. "
                 "The empire contains many provinces and cities. Roman roads are routes that connected settlements across the empire."),
    },
    "physics": {
        "title": "Physics: Gravity",
        "text": ("Gravity is the force that attracts objects with mass. "
                 "Earth is a planet with a solid surface and an atmosphere. "
                 "Earth contains water and many forms of life. "
                 "Gravity causes objects to accelerate toward one another."),
    },
    "computer_science": {
        "title": "Computer Science: Machine Learning",
        "text": ("Machine learning is a method for finding patterns in data. "
                 "Supervised learning uses labeled examples to train a model. "
                 "A neural network contains layers of connected units. "
                 "Training data provides examples for the learning process."),
    },
}
SAMPLE = SAMPLES["biology"]["text"]


def question_subject(subject: str) -> str:
    """Lower a leading article after a question word or auxiliary."""
    return re.sub(r"^(The|A|An)\s+", lambda m: m.group(1).lower() + " ", subject)


def do_auxiliary(subject: str) -> str:
    """Choose a basic do-support auxiliary from the final subject token."""
    final_word = re.sub(r"[^A-Za-z]", "", subject.split()[-1]).lower() if subject.split() else ""
    singular_s = {"photosynthesis", "analysis", "physics", "chemistry", "biology", "news", "gas", "osmosis"}
    plural = final_word.endswith("s") and not final_word.endswith(("ss", "us")) and final_word not in singular_s
    return "do" if plural else "does"


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if len(s.split()) >= 5]


def heuristic_tags(sentence: str) -> list[tuple[str, str]]:
    """Lightweight transparent POS approximation (not a trained tagger)."""
    words = re.findall(r"[A-Za-z][A-Za-z'-]*|\d+", sentence)
    out = []
    for i, w in enumerate(words):
        low = w.lower()
        tag = "NOUN"
        if low in {"is", "are", "was", "were", "has", "have", "do", "does", "can", "occurs", "provides", "absorbs", "convert", "converts", "take", "takes", "release", "releases"} or low.endswith(("ed", "ing")):
            tag = "VERB"
        elif low.endswith("ly"):
            tag = "ADV"
        elif low.endswith(("ous", "ive", "al", "ic", "able")):
            tag = "ADJ"
        elif w[0].isupper() and i > 0:
            tag = "PROPN"
        out.append((w, tag))
    return out


def linguistic_snapshot(sentence: str) -> tuple[str, str, str]:
    """Expose simple NLP signals without presenting them as model predictions."""
    tags = heuristic_tags(sentence)
    pos = ", ".join(f"{word}/{tag}" for word, tag in tags[:24])
    # Capitalization-based entity candidates are intentionally conservative.
    entities = list(dict.fromkeys(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", sentence)))
    ner = ", ".join(entities) if entities else "No capitalization-based entity candidate"
    relation = "Subject and relation cues are estimated from word order and verb patterns."
    return pos, ner, relation


def extract_candidates(text: str) -> list[tuple[str, str, str, str]]:
    """Return (answer, question, evidence, template) from factual patterns."""
    candidates: list[tuple[str, str, str, str]] = []
    for sentence in split_sentences(text):
        patterns = [
            (r"^(.{2,90}?)\s+is\s+(?:the\s+)?(?:process|study|branch|form|type|movement|force|organ|system|method|theory)\s+(?:by\s+which\s+)?(.+)$", "definition"),
            (r"^(.{2,70}?)\s+(is|are|was|were)\s+(.+)$", "is_are"),
            (r"^(.{2,100}?)\s+(?:occurs|takes place|happens)\s+(?:mainly\s+)?(?:in|at|during|when)\s+(.+)$", "where_when"),
            (r"^(.{2,100}?)\s+(?:absorbs|provides|produces|releases|contains|uses|requires|causes|results in)\s+(.+)$", "verb"),
            (r"^(.{2,100}?)\s+(?:take in|takes in|convert|converts)\s+(.+)$", "verb"),
        ]
        matched = False
        for pat, kind in patterns:
            m = re.match(pat, sentence, flags=re.I)
            if not m:
                continue
            if kind == "is_are":
                subject = m.group(1).strip(" ,;")
                copula = m.group(2).lower()
                obj = m.group(3).strip(" .;,")
            else:
                subject, obj = m.group(1).strip(" ,;"), m.group(2).strip(" .;,")
            if len(subject.split()) > 12 or len(obj.split()) < 2 or len(obj.split()) > 28:
                continue
            subject = question_subject(subject)
            if kind == "definition":
                q, answer = f"What is {subject}?", sentence[len(subject):].strip(" .")
            elif kind == "where_when":
                q, answer = f"Where or when {do_auxiliary(subject)} {subject} occur?", obj
            elif kind == "is_are":
                q, answer = f"What {copula} {subject}?", obj
            else:
                verb_match = re.search(r"\b(absorbs|provides|produces|releases|contains|uses|requires|causes|results in|take in|takes in|convert|converts)\b", sentence, re.I)
                verb = verb_match.group(1).lower() if verb_match else "does"
                base_verbs = {
                    "converts": "convert", "absorbs": "absorb", "provides": "provide",
                    "produces": "produce", "releases": "release", "contains": "contain",
                    "uses": "use", "requires": "require", "causes": "cause", "results in": "result in",
                    "takes in": "take in", "take in": "take in",
                }
                if verb in base_verbs:
                    q = f"What {do_auxiliary(subject)} {subject} {base_verbs[verb]}?"
                else:
                    q = f"What {do_auxiliary(subject)} {subject} do?"
                answer = obj
            candidates.append((answer, q, sentence, kind))
            matched = True
            break
        # Location and reason fallbacks grounded in explicit prepositions.
        if not matched:
            m = re.match(r"^(.{2,100}?)\s+(?:occurs|is found|are found)\s+(?:mainly\s+)?in\s+(.+)$", sentence, re.I)
            if m:
                candidates.append((m.group(2).rstrip(" ."), f"Where does {m.group(1)} occur?", sentence, "location"))
    # Preserve order and discard near-duplicates.
    seen: set[str] = set(); result = []
    for row in candidates:
        key = row[1].lower()
        if key not in seen:
            seen.add(key); result.append(row)
    return result


def score(question: str, answer: str, evidence: str) -> tuple[int, int, int, str]:
    qwords = set(re.findall(r"[a-z]+", question.lower()))
    ewords = set(re.findall(r"[a-z]+", evidence.lower()))
    overlap = len(qwords & ewords) / max(1, len(qwords))
    relevance = min(5, max(1, round(2 + 3 * overlap)))
    grammaticality = 4 if question.endswith("?") and question[:1].isupper() and len(question.split()) >= 4 else 2
    answerability = 5 if answer and answer.lower().strip(" .") in evidence.lower() else 3
    note = "Heuristic indicators only; human review is recommended."
    return relevance, grammaticality, answerability, note


def rule_generate(text: str, limit: int = 12) -> list[Question]:
    rows = []
    for answer, q, evidence, template in extract_candidates(text)[:limit]:
        r, g, a, note = score(q, answer, evidence)
        rows.append(Question(q, answer, evidence, "Rule-based", r, g, a, f"Template: {template}. {note}"))
    return rows


@lru_cache(maxsize=2)
def load_question_generation_pipeline(model_name: str):
    """Load each selected model once per running app process."""
    try:
        from transformers import pipeline
    except ImportError as exc:
        raise RuntimeError("Transformer mode needs the optional packages. Install requirements-transformers.txt.") from exc
    return pipeline("text2text-generation", model=model_name)


def transformer_generate(text: str, limit: int = 8, model_name: str = "iarfmoose/t5-base-question-generator") -> list[Question]:
    generator = load_question_generation_pipeline(model_name)
    output = generator("generate questions: " + text[:2500], max_new_tokens=limit * 40, num_return_sequences=1)
    raw = output[0]["generated_text"]
    questions = [q.strip(" -\n") for q in re.split(r"\s*\?\s*", raw) if len(q.split()) >= 4]
    sentences = split_sentences(text)
    rows = []
    for q in questions[:limit]:
        q = q.rstrip("?.") + "?"
        # Evidence is shown for review; answer is deliberately not fabricated by the generator.
        evidence = max(sentences, key=lambda s: len(set(q.lower().split()) & set(s.lower().split())), default="")
        r, g, a, note = score(q, "", evidence)
        rows.append(Question(q, "Review and enter answer", evidence, f"Transformer ({model_name})", r, g, 2, note + " Answer field requires human validation."))
    return rows


PAGE = """<!doctype html><html><head><meta charset=utf-8><title>EduQG</title><style>
body{font:16px system-ui;background:#f4f6fb;color:#14213d;margin:0}main{max-width:1000px;margin:38px auto;padding:0 22px}h1{font-size:36px;margin-bottom:4px}.sub{color:#516078}form,.item{background:white;padding:22px;border-radius:12px;margin:18px 0;box-shadow:0 2px 12px #14213d12}textarea{width:100%;height:210px;border:1px solid #cbd3e1;border-radius:8px;padding:12px;font:15px system-ui;box-sizing:border-box}select,button{padding:10px 14px;border-radius:7px;border:1px solid #bcc7d8;font:inherit;margin:8px 8px 0 0}button{background:#2454a6;color:white;border:0;cursor:pointer}.meta{color:#68758b;font-size:14px}.answer{color:#146356}.scores{font-size:14px;color:#515d70}details{margin-top:10px} .warn{background:#fff4d7;padding:12px;border-radius:8px}
</style></head><body><main><h1>EduQG</h1><div class=sub>Automatic question generation from educational text</div>
<form method=post><label for=sample><b>Choose a sample passage (optional)</b></label><br><select id=sample name=sample onchange="if(this.value){location.href='/?sample='+this.value}">__SAMPLE_OPTIONS__</select>
<p><label for=text><b>Or paste/edit any educational passage</b></label><br><textarea id=text name=text>__TEXT__</textarea></p>
<label>Generation method </label><select name=method><option value=rules __RULESEL__>Rule-based baseline</option><option value=transformer __TRANSSEL__>T5 transformer (optional)</option></select>
<button type=submit>Generate questions</button></form>
__ERROR__ __RESULT__
<p class=meta>Score indicators use transparent text heuristics. Review questions and answers before classroom use.</p></main></body></html>"""


def render(text: str, method: str = "rules", questions: Iterable[Question] = (), error: str = "", sample_key: str = "") -> str:
    cards = []
    for q in questions:
        pos, ner, relation = linguistic_snapshot(q.evidence)
        cards.append(f"<section class=item><h3>{html.escape(q.question)}</h3><p class=answer><b>Answer:</b> {html.escape(q.answer)}</p><p class=meta><b>{html.escape(q.method)}</b></p><p class=scores>Relevance {q.relevance}/5 · Grammaticality {q.grammaticality}/5 · Answerability {q.answerability}/5</p><details><summary>Evidence, NLP signals, and scoring note</summary><p><b>Evidence:</b> {html.escape(q.evidence)}</p><p><b>POS approximation:</b> {html.escape(pos)}</p><p><b>Entity candidates:</b> {html.escape(ner)}</p><p><b>Relation:</b> {html.escape(relation)}</p><p>{html.escape(q.explanation)}</p></details></section>")
    sample_options = ['<option value=""' + (" selected" if not sample_key else "") + '>Choose a sample…</option>']
    for key, sample in SAMPLES.items():
        selected = " selected" if key == sample_key else ""
        sample_options.append(f'<option value="{html.escape(key)}"{selected}>{html.escape(sample["title"])}</option>')
    return (PAGE.replace("__TEXT__", html.escape(text))
            .replace("__SAMPLE_OPTIONS__", "".join(sample_options))
            .replace("__RULESEL__", "selected" if method == "rules" else "")
            .replace("__TRANSSEL__", "selected" if method == "transformer" else "")
            .replace("__ERROR__", f'<p class="warn">{html.escape(error)}</p>' if error else "")
            .replace("__RESULT__", "".join(cards) if cards else ""))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        sample_key = parse_qs(urlparse(self.path).query).get("sample", [""])[0]
        if sample_key == "1":  # Keep the original sample URL working.
            sample_key = "biology"
        text = SAMPLES.get(sample_key, {}).get("text", "Paste educational text here.")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(render(text, sample_key=sample_key if sample_key in SAMPLES else "").encode())
    def do_POST(self):
        data = parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
        text = data.get("text", [""])[0][:12000]; method = data.get("method", ["rules"])[0]
        sample_key = data.get("sample", [""])[0]
        try:
            questions = rule_generate(text) if method == "rules" else transformer_generate(text)
            if not questions and method == "rules":
                error = "No supported rule patterns matched this passage. Try a clear definition or factual sentence, or choose the transformer option."
            else:
                error = "No questions were generated. Try a longer or clearer passage." if not questions else ""
        except Exception as exc:
            questions, error = [], str(exc)
        body = render(text, method, questions, error, sample_key).encode()
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *_): pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Launch the local browser interface")
    parser.add_argument("--text", help="Generate questions from a passage and print JSON")
    parser.add_argument("--method", choices=["rules", "transformer"], default="rules")
    parser.add_argument("--model", default="iarfmoose/t5-base-question-generator")
    args = parser.parse_args()
    if args.serve:
        print("EduQG is available at http://127.0.0.1:8000")
        HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
    else:
        text = args.text or SAMPLE
        items = rule_generate(text) if args.method == "rules" else transformer_generate(text, model_name=args.model)
        print(json.dumps([asdict(q) for q in items], indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
