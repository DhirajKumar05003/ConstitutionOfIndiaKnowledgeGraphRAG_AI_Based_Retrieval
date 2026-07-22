# LinkedIn post draft

Feel free to edit before posting — swap in a screen recording/GIF of the app and your repo link.

---

# LinkedIn post draft

Feel free to edit before posting — swap in a screen recording/GIF of the app and your repo link. This version leads with a civic-awareness angle rather than the pure tech pitch. Keep the framing to "know your rights" / "civic literacy" — deliberately avoids naming any specific protest, party, or government action, so it reads as informative rather than partisan.

---

📖 Most of us have heard of "fundamental rights" our whole lives. Far fewer of us have actually *read* them.

In a moment where rights, freedoms, and civic duties are being talked about everywhere — on the news, on campus, at the dinner table — I think the most useful thing any of us can do isn't pick a side of a debate. It's go back to the source document and actually understand it.

So I built something to make that easier.

🇮🇳 **ConstiGraph** — an AI assistant that lets you ask plain-language questions about the Constitution of India and get answers grounded directly in the text, with citations you can verify yourself.

Ask it things like:
🔹 "Can the government restrict freedom of speech, and under what conditions?"
🔹 "What protections exist against arbitrary arrest?"
🔹 "Is untouchability legally abolished, and how is it enforced?"

And it doesn't just answer — it shows you *which* Article and Clause the answer came from, on a live, explorable graph of the Constitution's actual structure.

**Why I think this matters right now:**
Rights are only as strong as the number of people who know they have them. A citizen who's actually read Article 19 or Article 21 is harder to mislead — by anyone, on any side. Civic literacy isn't a political stance. It's a form of self-defense.

**How it works, for the technically curious:**
Instead of the usual "chunk a PDF, dump it in a vector store" approach, I kept the Constitution's real structure — Part → Article → Clause → Subclause — as a knowledge graph. Every question runs through a full pipeline: query analysis → a retrieval planner that chooses between direct graph lookup and hybrid graph+vector search → evidence fusion → reranking → generation → a separate faithfulness-verification step that double-checks the answer's own citations before showing it to you. Built with Python, LangGraph, the Groq API, NetworkX, and Streamlit.

**Being upfront about its limits** (because a tool like this should never overstate its own authority):
⚠️ It currently covers Parts I–III of the Constitution (fundamental rights, citizenship, the Union) — not the full document yet.
⚠️ It has no case law or judicial precedent built in — real constitutional interpretation leans heavily on decades of court rulings this tool doesn't know.
⚠️ It's an educational aid, not legal advice, and never should be treated as a replacement for a lawyer, a teacher, or your own reading of the text.

**Where I want to take it:**
🎯 Extend the graph to cover the full Constitution, including Directive Principles and the Schedules
🎯 Add a "landmark judgments" layer so people can see how courts have actually interpreted these Articles over time
🎯 Multi-language support, so this isn't only useful to English speakers
🎯 A student mode with quizzes and simplified explainers, built for exam prep and classroom use

If you're a student, a first-time voter, or just someone who's been meaning to actually read the document that defines your rights — I'd love for you to try it and tell me what's missing.

Code (open source, with tests + architecture diagram): <your GitHub link>
Try it: <your demo link, if hosted>

Would love feedback — especially from law students, educators, or anyone who's thought about how AI can make civic knowledge more accessible instead of more confusing.

#Constitution #CivicLiteracy #KnowYourRights #AI #LangGraph #OpenSource #LegalTech #Students #India

---

## Shorter / punchier alternate (if you want something more scannable)

📖 How many of us have actually *read* our fundamental rights, instead of just hearing about them?

With so much conversation right now about rights and freedoms, I built ConstiGraph — an AI tool that answers plain-language questions about the Constitution of India, grounded in the actual text, with citations you can check yourself. Not to take a side in any debate — to make sure more of us can actually read the source material for ourselves.

Ask it "What does Article 19 protect?" or "What happens if someone is arrested?" and it answers from a knowledge graph of the real document, shows you exactly which Article/Clause it used, and even runs a second AI check to verify its own citations before showing you the answer.

Built with Python, LangGraph, Groq, and Streamlit. Open source, with an honest limitations section — it currently covers Parts I–III, has no case law yet, and is an educational aid, never a substitute for legal advice.

Next up: full Constitution coverage, landmark judgments, multi-language support, and a student/exam-prep mode.

If you're a student or a first-time voter, give it a try — knowing your rights is step one.

<your GitHub link>

#Constitution #CivicLiteracy #KnowYourRights #AI #OpenSource #Students

---

**Posting note:** pick one version (not both), swap in your real links, and consider adding a 20–30s screen recording of you asking it a question — civic-tech posts with a quick demo clip tend to land better than text alone.

