import time
import re
import copy
import os
import json
import asyncio
import hashlib
from typing import List, Literal, Dict, Set
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from src.state import DecisionStep, AgentState, CanonicalSkill
from src.vectorstore import  vstore_rag, vstore_cache
from src.utils import (extract_text_from_pdf, normalize_text, contains_token,
                       get_text_from_source)


# MODELS
fast_llm = ChatOllama(model="llama3.2:3b", temperature=0)
smart_llm = ChatOllama(model="llama3.1:8b", temperature=0, timeout=30, max_tokens=250)


async def input_router_node(state: AgentState):
    print(f"Avvio estrazione parallela per {len(state['available_sources'])} sorgenti.")
    return state






class SkillExtraction(BaseModel):
    skills: List[str] = Field(description="List of technologies, tools, or methodologies.")


async def extract_candidate_skills_from_source_node(state: dict) -> dict:
    start = time.perf_counter()
    source_id = state.get("source_id")
    source_data = state.get("source_data")

    if not source_id or not source_data:
        return {}

    path = source_data["path"]

    print(f"[EXTRACT] Processing source: {source_id} ({source_data['type']})")
    text = get_text_from_source(path)

    if not text:
        print(f"[WARN] No text extracted from {source_id}")
        return {"candidate_skills": []}


    text_hash = hashlib.md5(text.encode()).hexdigest()

    print(f"[DEBUG] Ricerca Cache - Source: {source_id}, Hash: {text_hash}")

    existing_docs = vstore_cache.similarity_search(
        query=f"skills_cache_{source_id}",
        k=1,
        filter={
            "$and": [
                {"source_id": {"$eq": source_id}},
                {"file_hash": {"$eq": text_hash}},
                {"type": {"$eq": "extracted_skills"}}
            ]
        },
    )

    if existing_docs:
        print(f"[CACHE HIT] Skill già estratte per {source_id}")
        extracted_names = json.loads(existing_docs[0]["content"])

        cached_skills = [
            {"skill": name, "source": source_id}
            for name in extracted_names
        ]

        return {
            "candidate_skills": cached_skills,
            "decisions": [DecisionStep(
                step=f"extraction_{source_id}",
                rule="Cache Hit Retrieval",
                inputs_used=[path],
                output=f"Loaded {len(cached_skills)} skills from cache"
            )]
        }

    print(f"[CACHE MISS] Calling LLM for {source_id}")

    llm = smart_llm
    structured_llm = llm.with_structured_output(SkillExtraction)

    prompt = f"""
        You are an expert technical recruiter. Extract explicitly mentioned technologies, tools, or methodologies from the text.
        Source Type: {source_data['type']}

        Rules:
        - Use only what is written, no inference.
        - Extract entities like programming languages, frameworks, databases, and DevOps tools.

        TEXT:
        {text}
        """

    try:
        response = await structured_llm.ainvoke(prompt)
        raw_skills_list = response.skills
    except Exception as e:
        print(f"Error during LLM extraction for {source_id}: {e}")
        raw_skills_list = []

    extracted_skills = [
        {"skill": s, "source": source_id}
        for s in raw_skills_list
    ]

    extracted_names = [s["skill"] for s in extracted_skills]

    skill_doc = Document(
        page_content=json.dumps(extracted_names),
        metadata={
            "source_id": source_id,
            "file_hash": text_hash,
            "type": "extracted_skills",
            "extracted_at": time.time()
        }
    )

    vstore_cache.add_documents([skill_doc])

    new_skills = [{"skill": name, "source": source_id} for name in extracted_names]

    end = time.perf_counter()
    print(f"[NODE END] {source_id} extraction completed in {end - start:.2f}s")

    return {
        "candidate_skills": new_skills,
        "decisions": [DecisionStep(
            step=f"extraction_{source_id}",
            rule="LLM Extraction & DB Storage",
            inputs_used=[path],
            output=f"Extracted and cached {len(new_skills)} skills"
        )]
    }








class TriageOutput(BaseModel):
    decision: str = Field(description="Decisione finale: 'ACCEPT' o 'REJECT'")
    reasoning: str = Field(description="Breve spiegazione della scelta")


async def triage_job_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] triage_job_node")

    candidate_skills = state.get("candidate_skills", [])
    skill_names = [s['skill'] for s in candidate_skills]
    candidate_summary = ", ".join(skill_names)

    job_text = state.get("job_raw_text", "")

    prompt = f"""
    Valuta la compatibilità macroscopica tra candidato e posizione lavorativa.

    REGOLE:
    - Se il candidato possiede almeno alcune hard skill rilevanti per il settore → ACCEPT
    - Se le competenze sono totalmente slegate dalla posizione → REJECT
    - Ignora soft skill.

    CANDIDATO SKILLS:
    {candidate_summary}

    JOB DESCRIPTION:
    {job_text}
    """

    try:
        structured_llm = fast_llm.with_structured_output(TriageOutput)
        result = await structured_llm.ainvoke(prompt)
        triage = result.decision.upper()
    except Exception as e:
        print(f"[WARN] Triage LLM failed: {e}. Fallback to string matching.")
        triage = "REJECT"

    if triage == "REJECT" and skill_names:
        job_lower = job_text.lower()
        overlap_count = 0
        for s_name in skill_names:
            norm_s = normalize_text(s_name)
            if norm_s and norm_s in job_lower:
                overlap_count += 1

        if overlap_count >= 2:
            print(f"[TRIAGE] Fallback triggered: {overlap_count} matches found. Overriding REJECT to ACCEPT.")
            triage = "ACCEPT"

    decision_step = DecisionStep(
        step="triage",
        rule="LLM analysis with heuristic string-match fallback",
        inputs_used=["candidate_skills", "job_raw_text"],
        output=triage
    )

    end = time.perf_counter()
    print(f"[NODE END] triage_job_node - decision={triage} - {end - start:.2f}s")


    return {
        "triage_decision": triage,
        "decisions": [decision_step]
    }








class JobRequirement(BaseModel):
    skill: str = Field(description="Nome della tecnologia o competenza.")
    search_keywords: List[str] = Field(default_factory=list, description="Termini tecnici puri contenuti nel requisito")
    category: Literal["DOMAIN_HARD_SKILLS", "DOMAIN_SOFT_SKILLS"] = Field(
        description="Categoria funzionale della competenza."
    )

class JobRequirementsOutput(BaseModel):
    requirements: List[JobRequirement]


async def extract_job_requirements_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] extract_job_requirements_node")

    job_text = state.get("job_raw_text", "")


    structured_llm = fast_llm.with_structured_output(JobRequirementsOutput)

    prompt = f"""
    Analizza la Job Description ed estrai i requisiti necessari.

    REGOLE:
    1. NUCLEO NOMINALE: Estrai solo il nome proprio (es. "Python", "Team Working"). 
       Elimina "esperienza in", "conoscenza di", ecc.
    2. ESPLOSIONE: Se trovi "Java o Kotlin", crea due entità separate.
    3. CATEGORIE:
   - DOMAIN_HARD_SKILLS: Tech stack, linguaggi, tool, certificazioni, framework e database. DEVI separare ogni singola tecnologia in un requisito indipendente.
     REGOLA FONDAMENTALE: Estrai OGNI SINGOLA TECNOLOGIA in un requisito separato e indipendente. Non raggruppare MAI più strumenti, concetti o tecnologie sotto un'unica voce, anche se nel testo sono scritti nella stessa frase (es. menzioni di tecnologie "preferite", "alternative" o "correlate" devono diventare requisiti atomici a sé stanti).
   - DOMAIN_SOFT_SKILLS: Mindset, attitudini, modalità di interazione e conoscenza di lingue straniere (es. Inglese/English).
    4. KEYWORDS: Per ogni requisito estratto, DEVI sempre fornire almeno una parola 
                 chiave nel campo 'search_keywords'. Se non ne trovi di specifiche,
                  inserisci il nome della skill stessa.

    TESTO:
    {job_text}
    """

    try:
        response = await structured_llm.ainvoke(prompt)
        raw_reqs = response.requirements
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        raw_reqs = []

    role_reqs = [
        {
            "skill": r.skill,
            "category": r.category,
            "search_keywords": r.search_keywords,
            "source": "job"
        }
        for r in raw_reqs if r.category == "DOMAIN_HARD_SKILLS"
    ]

    soft_reqs = [
        {
            "skill": r.skill,
            "category": r.category,
            "search_keywords": r.search_keywords,
            "source": "job"
        }
        for r in raw_reqs if r.category == "DOMAIN_SOFT_SKILLS"
    ]

    decision = DecisionStep(
        step="extract_requirements",
        rule="Structured Extraction (Hard vs Soft skills)",
        inputs_used=["job_raw_text"],
        output=f"Extracted {len(role_reqs)} hard and {len(soft_reqs)} soft skills"
    )

    end = time.perf_counter()
    print(f"[NODE END] extract_job_requirements_node - {end - start:.2f}s")

    return {
        "role_requirements": role_reqs,
        "soft_requirements": soft_reqs,
        "decisions": [decision]
    }










class SkillMatch(BaseModel):
    status: str = Field(
        description="Lo stato della competenza. Deve essere 'semantic_match', 'partial', o 'missing'."
    )
    source: str = Field(
        description="La fonte dell'analisi. Deve essere sempre la stringa 'llm'.",
        default="llm"
    )

class SkillGapOutput(BaseModel):
    analysis: Dict[str, SkillMatch] = Field(
        description="Mappa del requisito del job verso i dettagli di match."
    )


async def skill_gap_analysis_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] skill_gap_analysis_node")

    role_reqs = state.get("role_requirements", [])
    cand_skills = state.get("candidate_skills", [])

    if not role_reqs:
        return {"skill_gap": {}, "match_score": 0.0}


    gap_results = {}
    remaining_reqs = []
    remaining_keywords = {}

    canonical = state.get("canonical_skills", [])
    candidate_semantics = {}
    job_semantics = {}

    for item in canonical:

        tags = set(
            t.lower().strip()
            for t in item.get("capability", [])
        )

        if item["source"] != "job":
            candidate_semantics[item["original"]] = tags
        else:
            job_semantics[item["original"]] = tags

    for req in role_reqs:
        req_name = req["skill"]
        found = False
        for cand in cand_skills:
            if contains_token(cand["skill"], req_name):
                status = "direct_match"
                gap_results[req_name] = {"status": status, "source": "token_match"}
                found = True
                break

        if not found:
            req_tags = job_semantics.get(req_name, set())

            semantic_overlap = False

            for cand_name, cand_tags in candidate_semantics.items():

                overlap = req_tags.intersection(cand_tags)

                if len(overlap) >= 1:
                    semantic_overlap = True
                    break

            if semantic_overlap:

                gap_results[req_name] = {
                    "status": "semantic_match",
                    "source": "semantic_tags"
                }

            else:

                remaining_reqs.append(req_name)

                remaining_keywords[req_name] = (
                    req["search_keywords"]
                    if req["search_keywords"]
                    else req_name
                )

            remaining_keywords[req_name] = req["search_keywords"] if req["search_keywords"] else req_name


    if remaining_reqs:
        cand_summary = ", ".join([s["skill"] for s in cand_skills])

        structured_llm = fast_llm.with_structured_output(SkillGapOutput)
        prompt = f"""
                Sei un recruiter tecnico senior. Confronta i requisiti mancanti del Job con le competenze estratte dal profilo del candidato per trovare equivalenze semantiche o sinonimi.

                REQUISITI DA VERIFICARE: 
                {remaining_reqs}

                KEYWORDS ASSOCIATE AI REQUISITI:  
                {remaining_keywords}

                COMPETENZE ESTRATTE DAL CANDIDATO: 
                {cand_summary}

                REGOLE DI VALUTAZIONE DIETRO LE QUINTE:
                - 'semantic_match': esiste una competenza equivalente o fortemente correlata.
                - 'partial': esiste una relazione debole.
                - 'missing': nessuna evidenza.

                REGOLE DI FORMATTAZIONE TASSATIVE PER L'OUTPUT JSON:
                1. Il dizionario 'analysis' inserito nell'output DEVE contenere come CHIAVI esclusivamente e testualmente le stringhe presenti nella lista 'REQUISITI DA VERIFICARE'.
                2. NON modificare le maiuscole/minuscole, NON aggiungere o togliere parole dalle chiavi. 
                3. Ad esempio, se nella lista c'è "Proficiency", "Expertise", "Knowledge", la chiave nel dizionario DEVE essere l'intera parola originale.
                4. Inserisci una ed una sola voce per ognuno dei requisiti richiesti.
                5. Il VALORE per ogni chiave deve essere un oggetto contenente "status" (il risultato) e "source" (sempre "llm").
                """

        try:
            llm_result = await structured_llm.ainvoke(prompt)

            cleaned_analysis = {}
            for req_name in remaining_reqs:
                llm_match = next(
                    (v for k, v in llm_result.analysis.items() if k.lower().strip() == req_name.lower().strip()), None)
                if llm_match:
                    cleaned_analysis[req_name] = {"status": llm_match.status, "source": llm_match.source}
                else:
                    cleaned_analysis[req_name] = {"status": "missing", "source": "llm"}

            gap_results.update(cleaned_analysis)
        except Exception as e:
            print(f"[WARN] LLM Gap Analysis failed: {e}")
            for r in remaining_reqs:
                if r not in gap_results:
                    gap_results[r] = {"status": "missing", "source": "error"}

    weights = {
        "direct_match": 1.0,
        "semantic_match": 0.8,
        "partial": 0.5,
        "missing": 0.0
    }
    total_points = sum(weights.get(v["status"], 0.0) for v in gap_results.values())
    match_score = round(total_points / len(role_reqs), 2) if role_reqs else 0.0

    matches_count = sum(1 for v in gap_results.values() if v["status"] in ["direct_match", "semantic_match"])

    decision = DecisionStep(
        step="skill_gap_analysis",
        rule="Hybrid deterministic & LLM semantic matching",
        inputs_used=["candidate_skills", "role_requirements"],
        output=f"Score: {match_score:.2f}, Matches: {matches_count}"
    )

    end = time.perf_counter()
    print(f"[NODE END] skill_gap_analysis_node - score={match_score} - {end - start:.2f}s")

    return {
        "skill_gap": gap_results,
        "match_score": match_score,
        "decisions": [decision]
    }














class CanonicalItem(BaseModel):
    original: str = Field(description="Nome originale della skill")
    retrieval_tokens: List[str] = Field(description= "SOLO parole che devono esistere nel testo di un CV")
    semantic_tags: List[str] = Field(description="Tag per ragionamenti dell'LLM")
    #capability: str = Field(description="Nome della macro-categoria (es. BACKEND_DEVELOPMENT)")
    #dimensions: List[str] = Field(description="Concetti atomici correlati (es. ['python', 'api'])")


class CanonicalList(BaseModel):
    mappings: List[CanonicalItem]


async def canonicalize_skills_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] canonicalize_skills_node")

    candidate_raw = state.get("candidate_skills", [])
    job_raw = state.get("role_requirements", [])

    unique_names = list({s["skill"] for s in (candidate_raw + job_raw) if s.get("skill")})

    if not unique_names:
        return {"canonical_skills": []}


    cache_file = "data/canonical_cache.json"
    canonical_cache = {}

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                canonical_cache = json.load(f)
        except Exception as e:
            print(f"[WARN] Impossibile caricare {cache_file}: {e}")

    current_run_mappings = {}
    missing_skills = []

    for skill in unique_names:
        key = skill.lower().strip()
        if key in canonical_cache:
            current_run_mappings[skill] = canonical_cache[key]
        else:
            missing_skills.append(skill)

    if missing_skills:
        print(f"  [LLM] Canonicalizzazione di {len(missing_skills)} nuove skill...")
        structured_llm = fast_llm.with_structured_output(CanonicalList)




        prompt = f"""
        You normalize skills for retrieval.
    
        For each skill produce:
        1) retrieval_tokens are EXACT SEARCH KEYS.
            - must be 1–2 words max
            - must be the minimal canonical form
            - must NOT contain adjectives (proficiency, experience, knowledge)
            - must NOT contain phrases
            - must NOT contain multi-word descriptions unless acronym
            - If a skill contains a known technology inside it, ALWAYS extract the technology as retrieval_token.
            - If a skill is a platform/service (Redshift, Databricks, Snowflake): keep original token as retrieval_token, do NOT abstract it
    
        2) semantic_tags:
        - short conceptual labels
        - maximum 3 tags
        - avoid generic tags (e.g. platform, library, etc.)
    
    
        Skills:
        {unique_names}
        """

        try:
            response = await structured_llm.ainvoke(prompt)

            for m in response.mappings:
                original_skill = m.original
                cache_key = original_skill.lower().strip()

                mapping_data = {
                    "retrieval_tokens": m.retrieval_tokens,
                    "semantic_tags": m.semantic_tags
                    }

                current_run_mappings[original_skill] = mapping_data
                canonical_cache[cache_key] = mapping_data

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(canonical_cache, f, indent=2, ensure_ascii=False)

            print(f"  [CACHE] Aggiornata con {len(response.mappings)} nuovi record.")


        except Exception as e:
            print(f"[ERROR] Canonicalization failed for missing skills: {e}")




    final_canonical: List[CanonicalSkill] = []

    for raw_item in (candidate_raw + job_raw):
        name = raw_item["skill"]
        if name in current_run_mappings:
            m = current_run_mappings[name]
            final_canonical.append({
                "original": name,
                "capability": m.get("semantic_tags", []),
                "dimensions": [d.lower() for d in m.get("retrieval_tokens", [])],
                "source": raw_item["source"]
            })

    decision = DecisionStep(
        step="canonicalize_skills",
        rule="Abstract capability mapping via LLM",
        inputs_used=["candidate_skills", "role_requirements"],
        output=f"Mapped {len(unique_names)} unique skills to {len(final_canonical)} entries"
    )

    end = time.perf_counter()
    print(f"[NODE END] canonicalize_skills_node - {end - start:.2f}s")

    return {
        "canonical_skills": final_canonical,
        "decisions": [decision]
    }









async def match_capabilities_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] match_capabilities_node")

    canonical_skills = state.get("canonical_skills", [])
    skill_gap = state.get("skill_gap", {})

    potential_rag_targets_map = {}

    for skill in canonical_skills:
        if skill.get("source") != "job":
            continue
        name = skill["original"]
        gap_info = skill_gap.get(name, {})
        gap_status = gap_info.get("status", "missing")
        gap_source = gap_info.get("source", "llm")

        if gap_status == "direct_match" and gap_source != "llm":
            continue

        tokens = skill.get("dimensions", [])
        semantic_tags = skill.get("capability", [])

        potential_rag_targets_map[name] = {
            "tokens": tokens,
            "semantic_tags": semantic_tags
        }


    decision = DecisionStep(
        step="capability_match",
        rule="Identify missing requirements within candidate's known capabilities",
        inputs_used=["canonical_skills", "skill_gap"],
        output=f"Targeting {len(potential_rag_targets_map)} skills for RAG deep-dive"
    )

    end = time.perf_counter()
    print(f"[NODE END] match_capabilities_node - Found {len(potential_rag_targets_map)} targets - {end - start:.2f}s")

    return {
        "potential_rag_targets": potential_rag_targets_map,
        "decisions": [decision]
    }













async def analysis_path_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] analysis_path_node")

    match_score = state.get("match_score", 0.0)
    triage = state.get("triage_decision", "REJECT")
    rag_targets = state.get("potential_rag_targets", {})

    if triage == "REJECT":
        path = "LIGHT"
        rule = "Triage was REJECT: skip deep analysis."

    elif len(rag_targets) > 0:
        path = "DEEP"
        rule = f"Found {len(rag_targets)} actionable gaps within candidate's capabilities."

    elif match_score < 0.2:
        path = "LIGHT"
        rule = "Match score too low and no relevant capabilities to investigate."

    else:
        path = "LIGHT"
        rule = "Sufficient match or no relevant targets for RAG investigation."

    decision = DecisionStep(
        step="analysis_path_selection",
        rule=rule,
        inputs_used=["triage_decision", "match_score", "potential_rag_targets"],
        output=path
    )

    end = time.perf_counter()
    print(f"[NODE END] analysis_path_node - path={path} - {end - start:.2f}s")

    return {
        "analysis_path": path,
        "decisions": [decision]
    }









async def light_report_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] light_report_node")

    match_score = state.get("match_score", 0.0)
    skill_gap = state.get("skill_gap", {})

    missing_skills = [s for s, status in skill_gap.items() if status["status"] == "missing"]
    partial_skills = [s for s, status in skill_gap.items() if status["status"] == "partial"]

    report_lines = [
        "## Technical Match Analysis (Light Mode)",
        f"**Match Score Complessivo:** `{match_score:.2f}`\n",
        "> **Nota:** È stata eseguita un'analisi veloce. Dato il punteggio o la mancanza di evidenze correlate, non è stata attivata l'investigazione profonda (RAG).",
        "\n### Riepilogo Gap Tecnici"
    ]

    if missing_skills:
        report_lines.append(f"- **Competenze Mancanti:** {', '.join(missing_skills)}")

    if partial_skills:
        report_lines.append(f"- **Competenze Parziali/Dubbie:** {', '.join(partial_skills)}")

    if not missing_skills and not partial_skills:
        report_lines.append("- Nessun gap critico rilevato nello scan iniziale.")

    report_lines.append("\n---\n")

    decision = DecisionStep(
        step="light_report",
        rule="Fast-path report generation",
        inputs_used=["match_score", "skill_gap"],
        output=f"Light report generated (score: {match_score:.2f})"
    )

    end = time.perf_counter()
    print(f"[NODE END] light_report_node - {end - start:.2f}s")

    return {
        "report": "\n".join(report_lines),
        "decisions": [decision]
    }














async def build_rag_corpus_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] build_rag_corpus_node")

    available_sources = state.get("available_sources", {})
    if not available_sources:
        return {"rag_ready": False}


    combined_content = ""
    for s_id, s_data in available_sources.items():
        path = s_data["path"]
        mtime = os.path.getmtime(path)
        fsize = os.path.getsize(path)
        combined_content += f"|{path}_{mtime}_{fsize}"

    global_hash = hashlib.md5(combined_content.encode()).hexdigest()


    if vstore_rag.is_collection_valid(global_hash):
        print("[RAG] Vectorstore is up-to-date. Reusing existing collection.")
        decision = DecisionStep(
            step="rag_corpus_build",
            rule="Global hash match",
            inputs_used=list(available_sources.keys()),
            output="reused_existing_db"
        )
        return {"rag_ready": True, "decisions": [decision]}


    print("[RAG] Change detected or DB missing. Rebuilding vectorstore...")

    vstore_rag.reset_db()

    all_documents = []

    for source_id, source_data in available_sources.items():
        path = source_data["path"]

        print(f"[EXTRACT] Processing source: {source_id} ({source_data['type']})")
        text = get_text_from_source(path)

        chunks = vstore_rag.create_chunks(text, metadata={
            "source_id": source_id,
            "type": source_data["type"],
            "global_hash": global_hash
        })
        all_documents.extend(chunks)

    if all_documents:
        vstore_rag.add_documents(all_documents)
        vstore_rag.save_collection_metadata(global_hash)
        status = "created_new_db"
    else:
        status = "failed_no_content"

    decision = DecisionStep(
        step="rag_corpus_build",
        rule="Atomic rebuild due to hash mismatch",
        inputs_used=list(available_sources.keys()),
        output=status
    )

    end = time.perf_counter()
    print(f"[NODE END] build_rag_corpus_node - {status} - {end - start:.2f}s")

    return {
        "rag_ready": True,
        "decisions": [decision]
    }










async def retrieve_explanatory_context_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] retrieve_explanatory_context_node (Hybrid Mode)")

    if not state.get("rag_ready"):
        print("[ERROR] RAG corpus not ready, skipping retrieval.")
        return {}

    targets = state.get("potential_rag_targets", {})

    if not targets:
        return {"explanatory_context": {}}

    vectorstore_rag = vstore_rag.get_vectorstore()

    db = vectorstore_rag.get(include=["documents", "metadatas"])

    documents = db.get("documents", [])
    metadata = db.get("metadatas", [])

    associated_elements = []
    for text, meta in zip(documents, metadata):
        associated_elements.append({
            "content": text,
            "metadata": meta
        })


    explanatory_context = {}
    total_evidence_found = 0

    for skill, data in targets.items():

        tokens = data.get("tokens", [])
        semantic_tags = data.get("semantic_tags", [])

        #print(f"  [SEARCH TARGET] {skill}")

        current_snippets = []
        keyword_docs = []

        if associated_elements and tokens:
            for doc in associated_elements:
                content = doc.get("content", "").lower()

                is_match = any(token.lower().strip() in content for token in tokens if token.strip())

                if is_match:
                    keyword_docs.append(doc)

        if keyword_docs:
            #print(f"    -> [LEVEL 1 HIT] Match testuale per i token: {tokens}")
            for d in keyword_docs:
                current_snippets.append({
                    "content": d.get("content", ""),
                    "metadata": d.get("metadata", "")
                })

        else:
            #print(f"    -> [LEVEL 2 FALLBACK] Nessun match esatto. Avvio Vector Search...")

            query_parts = [skill] + semantic_tags + tokens
            semantic_query = " ".join(query_parts)
            #print(f"       Query semantica: '{semantic_query}'")

            try:
                potential_docs = vstore_rag.similarity_search(semantic_query, k=3)
                for d in potential_docs:
                    current_snippets.append({
                        "content": d.get("content", ""),
                        "metadata": d.get("metadata", "")
                    })
            except Exception as e:
                print(f"[ERROR] Vector search fallita per {skill}: {e}")

        unique_snippets = []
        seen_content = set()
        for s in current_snippets:
            if s["content"] not in seen_content:
                unique_snippets.append(s)
                seen_content.add(s["content"])

        if unique_snippets:
            total_evidence_found += 1
            explanatory_context[skill] = {
                "snippets": unique_snippets[:3],
                "dimensions": tokens,
                "semantic_tags": semantic_tags
            }

    decision = DecisionStep(
        step="retrieve_explanatory_context",
        rule="Multi-stage cascading search (Lexical -> Semantic)",
        inputs_used=["potential_rag_targets_map"],
        output=f"Retrieved context for {len(targets)} targets"
    )

    end = time.perf_counter()
    print(f"[NODE END] retrieve_explanatory_context_node - {end - start:.2f}s")

    return {
        "explanatory_context": explanatory_context,
        "decisions": [decision]
    }












class SkillAudit(BaseModel):
    skill: str
    supported_dimensions: List[str] = Field(default_factory=list)
    justification: str
    verdict: Literal[
        "CONFERMATA",
        "PARZIALE",
        "NON TROVATA"
    ]
    evidence_quote: str

class BatchSkillAudit(BaseModel):
    audits: List[SkillAudit]


async def analyze_skills_batch(skills_data: List[dict]) -> List[SkillAudit]:

    payload = []

    for item in skills_data:
        evidence_text = "\n---\n".join(
            [s["content"][:350] for s in item["snippets"][:2]]
        )

        payload.append({
            "skill": item["skill"],
            "dimensions": item["dimensions"],
            "capability": item["semantic_tags"],
            "evidence": evidence_text
        })

    prompt = f"""
    You are a technical recruiter.

    For EACH skill:

    supported_dimensions:
    - must contain ONLY dimensions explicitly found in evidence
    - if nothing is found return []

    justification:
    - A brief justification in italian of why the skill is confirmed or not
    
    verdict:
    - Must be either: 
        - CONFERMATO: evidenza forte e diretta 
        - PARZIALE: almeno una evidenza concreta o almeno una dimensione confermata
        - NON TROVATO: nessuna evidenza, dimensione o citazione confermata
    
    evidence quote:
    - if CONFERMATO or PARZIALE:
        - extract a short quote (max 20 words)
        - explain why it supports the skill
    - if NON TROVATO:
      - leave evidence_quote empty

    RULE:
    Use capability to help you identify if a skill is present in another form or not

    Return structured output.

    DATA:
    {json.dumps(payload, ensure_ascii=False)}
    """

    structured_llm = smart_llm.with_structured_output(BatchSkillAudit)
    result = await structured_llm.ainvoke(prompt)

    return result.audits


async def deep_analysis_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] deep_analysis_node")

    skill_gap = copy.deepcopy(state.get("skill_gap", {}))
    context_map = state.get("explanatory_context", {})
    tech_reqs = state.get("role_requirements", [])

    tasks = []
    analysis_results: Dict[str, SkillAudit] = {}

    batch_input = []

    for req in tech_reqs:
        name = req["skill"]
        ctx = context_map.get(name, {})
        snippets = ctx.get("snippets", [])
        keywords = ctx.get("dimensions", [name])
        semantic_tags = ctx.get("semantic_tags", [])

        gap_data = skill_gap.get(name, {})

        current_gap_status = gap_data.get("status", "missing")
        current_gap_source = gap_data.get("source", "llm")


        if current_gap_status == "direct_match" and current_gap_source != "llm":
            continue

        if not snippets:
            analysis_results[name] = SkillAudit(
                skill=name,
                supported_dimensions=[],
                verdict="NON TROVATA",
                justification="Non trovato. Nessuna evidenza recuperata dal RAG.",
                evidence_quote=""
            )
            continue

        confirmed_keywords = []
        for kw in keywords:
            if any(kw.lower().strip() in s["content"].lower() for s in snippets):
                confirmed_keywords.append(kw)


        if confirmed_keywords:
            evidence = ""

            for kw in confirmed_keywords:
                for s in snippets:
                    content = s["content"]

                    idx = content.lower().find(kw.lower())

                    start_kw = max(0, idx - 100)
                    end_kw = min(len(content), idx + len(kw) + 100)

                    if kw.lower() in content.lower():
                        evidence = content[start_kw:end_kw]
                        break

                if evidence:
                    break

            #print(f"  [LEVEL 1 MATCH] '{name}' verificata deterministicamente tramite keyword: {confirmed_keywords}")
            analysis_results[name] = SkillAudit(
                skill=name,
                supported_dimensions=confirmed_keywords,
                verdict="CONFERMATA",
                justification=f"Verificato deterministicamente tramite presenza esplicita nel testo delle keyword: {confirmed_keywords}.",
                evidence_quote=evidence
            )
            continue

        batch_input.append({
            "skill": name,
            "snippets": snippets,
            "dimensions": keywords,
            "semantic_tags": semantic_tags
        })

        #print(f"  [LEVEL 2 FALLBACK] '{name}' richiede l'intervento dell'LLM...")
        reduced_ctx = {
            "dimensions": keywords,
            "snippets": snippets[:2]
        }



    if batch_input:
        print(f"[BATCH AUDIT] Analizzando {len(batch_input)} skills insieme...")
        results = await analyze_skills_batch(batch_input)

        for r in results:
            analysis_results[r.skill] = r


    report_sections = ["\n## Deep Evidence Analysis\n"]

    job_reqs = [r["skill"] for r in state.get("role_requirements", []) if r.get("source") == "job"]
    if not job_reqs:
        job_reqs = list(skill_gap.keys())

    for name in job_reqs:
        audit = analysis_results.get(name)

        if name not in skill_gap:
            skill_gap[name] = {"status": "missing", "source": "llm"}

        current_status = skill_gap[name].get("status", "missing")
        current_source = skill_gap[name].get("source", "llm")

        if current_status == "direct_match" and current_source != "llm":
            report_sections.append(f"### {name} → **DIRECT_MATCH** (Screening Diretto)")
            report_sections.append(
                "*Analisi:* Match esatto confermato per prova diretta nei documenti o CV estratti. Nessuna ulteriore analisi RAG necessaria.\n")
            continue


        if audit is not None:
            justification_lower = audit.justification.lower() if audit.justification else ""

            verdict = audit.verdict.upper()

            has_dimensions = bool(audit.supported_dimensions)
            has_evidence = bool(audit.evidence_quote and audit.evidence_quote.strip())

            if verdict != "NON TROVATO" and not has_dimensions and not has_evidence:
                verdict = "NON TROVATO"

            negative_analysis = any(
                token in audit.justification.lower()
                for token in [
                    "non trovato",
                    "not found",
                    "non utilizza",
                    "assenza",
                    "missing"
                ]
            )

            if negative_analysis:
                verdict = "NON TROVATO"

            if verdict == "NON TROVATO":
                new_status = "missing"

            elif verdict == "PARZIALE":
                new_status = "partial"

            elif verdict == "CONFERMATA":
                if current_status == "semantic_match" and audit.supported_dimensions:
                    new_status = "direct_match"
                else:
                    new_status = "semantic_match"

            else:
                new_status = current_status

            skill_gap[name]["status"] = new_status
            skill_gap[name]["source"] = "rag_audit"

            dims_str = ", ".join(audit.supported_dimensions) if audit.supported_dimensions else 'Nessuna'

            report_sections.append(f"### {name} → **{new_status.upper()}** (Audit Evidenze)")
            report_sections.append(f"**Dimensioni confermate:** {dims_str}")
            report_sections.append(f"*Analisi:* {audit.justification}")
            report_sections.append(f"*Evidenze:* {audit.evidence_quote}\n")

        else:
            current_status = skill_gap[name].get("status", "missing")
            report_sections.append(f"### {name} → **{current_status.upper()}** (In base a screening iniziale)")
            report_sections.append(
                f"*Analisi:* Confermato dallo screening iniziale. Nessuna evidenza contraria o frammento integrativo estratto dal RAG.\n"
            )

    weights = {
        "direct_match": 1.0,
        "semantic_match": 0.8,
        "partial": 0.5,
        "missing": 0.0,
    }
    raw_score = sum(weights.get(v.get("status","missing"), 0.0) for v in skill_gap.values()) / len(skill_gap) if skill_gap else 0.0
    final_score = round((state.get("match_score", 0.0) * 0.6) + (raw_score * 0.4), 2)

    decision = DecisionStep(
        step="deep_analysis",
        rule="Evidence-based audit with anti-hallucination filter",
        inputs_used=["explanatory_context"],
        output=f"Analyzed {len(tasks)} skills, Final Score: {final_score}"
    )

    end = time.perf_counter()
    print(f"[NODE END] deep_analysis_node - {end - start:.2f}s")

    return {
        "report": state.get("report", "") + "\n".join(report_sections),
        "match_score": final_score,
        "skill_gap": skill_gap,
        "decisions": [decision]
    }


















class ProfessionalEval(BaseModel):
    skill: str

    status: str = Field(
        description="FOUND | PROBABLE | NO EVIDENCE"
    )

    confidence: str = Field(
        description="HIGH | AVERAGE | LOW"
    )

    evidence_quote: str = Field(
        description="Breve citazione dal CV che supporta la valutazione. Vuoto se non disponibile."
    )

    observation: str = Field(
        description="Spiegazione sintetica della valutazione."
    )


class ProfessionalReportOutput(BaseModel):
    evaluations: List[ProfessionalEval]


async def professional_requirements_report_node(state: AgentState) -> dict:
    start = time.perf_counter()
    print("[NODE START] professional_requirements_report_node")

    soft_reqs = state.get("soft_requirements", [])

    if not soft_reqs:
        decision = DecisionStep(
            step="professional_requirements_report",
            rule="Semantic check for soft requirements",
            inputs_used=[],
            output=f"There are no soft skills in the requirements"
            )
        return {"report": state.get("report", ""),
                "decision": decision,
                }

    cv_text = state.get("cv_raw_text", "")


    structured_llm = fast_llm.with_structured_output(
        ProfessionalReportOutput
    )

    prompt = f"""
    Sei un recruiter senior.
    
    Analizza il CV e valuta se i requisiti comportamentali richiesti
    sono supportati da evidenze presenti nel testo.
    
    REQUISITI:
    
    {[r["skill"] for r in soft_reqs]}
    
    CV:
    
    {cv_text[:10000]}
    
    REGOLE:
    
    1. FOUND
       - Evidenze forti e dirette.
       - Esperienze, progetti o risultati dimostrano chiaramente il requisito.
    
    2. PROBABLE
       - Il requisito non è esplicito ma può essere inferito
         da attività, responsabilità o progetti.
    
    3. NO EVIDENCE
       - Nessuna evidenza significativa.
    
    Per ogni requisito restituisci:
    
    - status
    - confidence
    - evidence_quote
    - observation
    
    IMPORTANTE:
    
    - Non inventare evidenze.
    - evidence_quote deve essere una citazione reale del CV.
    - Se non trovi una citazione usa stringa vuota.
    - Mantieni observation entro una frase.
    """

    try:
        response = await structured_llm.ainvoke(prompt)
        evals = response.evaluations

    except Exception as e:
        print(f"[ERROR] Professional evaluation failed: {e}")

        evals = [
            ProfessionalEval(
                skill=req["skill"],
                status="NON EVIDENTE",
                confidence="BASSA",
                evidence_quote="",
                observation="Errore durante la valutazione."
            )
            for req in soft_reqs
        ]

    report_sections = [
        "\n## Professional & Behavioral Requirements",
        "*Nota:* Valutazione qualitativa separata dal match score tecnico.\n"
    ]

    for ev in evals:

        report_sections.append(
            f"### {ev.skill} → **{ev.status}** ({ev.confidence})"
        )

        report_sections.append(
            f"*Osservazione:* {ev.observation}"
        )

        if ev.evidence_quote:
            report_sections.append(
                f"*Evidenza:* {ev.evidence_quote}"
            )

        report_sections.append("")

    decision = DecisionStep(
        step="professional_requirements_report",
        rule="Global CV behavioral analysis",
        inputs_used=[
            "cv_raw_text",
            "soft_requirements"
        ],
        output=f"Evaluated {len(evals)} soft requirements"
    )

    end = time.perf_counter()

    print(
        f"[NODE END] professional_requirements_report_node - {end - start:.2f}s"
    )

    current_report = state.get("report", "")

    return {
        "report": current_report + "\n".join(report_sections),
        "decisions": [decision]
    }