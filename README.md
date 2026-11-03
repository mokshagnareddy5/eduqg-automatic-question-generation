# EduQG: Automatic Question Generation from Educational Text

EduQG is a small research prototype for generating factual study questions from textbook passages. It includes an interpretable rule based baseline and an optional T5 transformer mode. Each result includes an answer or review reminder, supporting evidence, and transparent heuristic indicators for relevance, grammaticality, and answerability.

## Run the browser demo

Requires Python 3.10 or newer. The rule based project uses only Python's standard library.

```powershell
python aqg_project.py --serve
```

Open `http://127.0.0.1:8000`, choose one of the Biology, History, Physics, or Computer Science examples if useful, or paste and edit any educational passage. The sample passages are optional examples; the app accepts user-provided text. The server listens only on the local computer.

## Project video

Presentation and live demo video: [Watch the project video on OneDrive](https://1drv.ms/v/c/a6b16236bddfc794/IQD0d1QMVKxsSqPG4rPH-n7kAXkrQG0YQvd4gdYrdvVYSe8?e=jrTLnT)

## Run from the command line

```powershell
python aqg_project.py --text "Photosynthesis is the process by which plants convert light into chemical energy."
```

The command prints JSON records containing each generated question, answer, evidence sentence, method, and score indicators.

## Optional transformer mode

Install compatible PyTorch and Transformers packages, then install `requirements-transformers.txt` and run:

```powershell
python aqg_project.py --serve
```

Choose **T5 transformer (optional)** in the browser. The first use downloads and loads the configured model, so it may take longer. While the app stays open, the loaded model is reused for later generations. Restarting the app clears that in-memory cache, although downloaded model files remain in the local Hugging Face cache. Set a different model using `--model` in CLI mode. Transformer output needs human answer validation; the demo shows the closest evidence sentence and does not claim that it is a verified answer.

## Method and limitations

The rule baseline uses sentence splitting, transparent lexical patterns, and simple question templates. It is not limited to the built-in samples, but it only generates questions when sentences match its supported patterns. Its POS tagger is a heuristic approximation, not a trained tagger. It will miss many sentence forms and can produce awkward questions, especially with complex clauses. The score values are debugging indicators, not benchmark metrics or human judgments. Review all generated questions before using them for teaching or assessment.

## Project files

- `aqg_project.py`: web interface, rule baseline, optional transformer integration, and CLI.
- `presentation/EduQG_Project_Presentation.pptx`: editable eight slide project presentation with speaker notes.
- `requirements-transformers.txt`: optional model dependencies.
- `report/EduQG_Final_Project_Report_A4.docx`: final A4 report in editable format.
