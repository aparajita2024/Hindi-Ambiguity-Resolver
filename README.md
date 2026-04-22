# 🇮🇳 Hindi Ambiguity Resolution System

> **An end-to-end NLP system that detects and resolves ambiguity in Hindi language — supporting both Devanagari script and Hinglish (Roman) input, powered by a rule-based engine and a 2-Billion parameter Transformer model.**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Features](#-features)
- [Ambiguity Types Covered](#-ambiguity-types-covered)
- [Project Structure](#-project-structure)
- [System Architecture](#-system-architecture)
- [2B Parameter Transformer Model](#-2b-parameter-transformer-model)
- [Rule-Based Database](#-rule-based-database)
- [Hinglish Input Support](#-hinglish-input-support)
- [Web Interface](#-web-interface)
- [Installation & Setup](#-installation--setup)
- [Running in Google Colab](#-running-in-google-colab)
- [Usage](#-usage)
- [Expected Output](#-expected-output)
- [Common Errors & Fixes](#-common-errors--fixes)
- [Technologies Used](#-technologies-used)
- [RAM & Performance Optimization](#-ram--performance-optimization)
- [Future Work](#-future-work)
- [License](#-license)

---

## 🔍 Overview

Ambiguity is one of the most challenging problems in Natural Language Processing. In Hindi, a single word like **कल** means both *yesterday* and *tomorrow*. The word **सोना** means both *to sleep* and *gold*. An idiom like **दिल टूट गया** literally means *the heart broke* but figuratively means *heartbreak*.

This project explicitly surfaces **both interpretations** of any ambiguous Hindi word, sentence, or idiom — rather than collapsing the ambiguity into a single dominant meaning.

### Key Numbers

| Metric | Value |
|--------|-------|
| Rule-based database entries | 60 |
| Ambiguity types | 6 |
| Hinglish → Hindi mappings | 100+ |
| Transformer parameters | ~2 Billion |
| Decoder layers | 24 |
| Attention heads (Q / KV) | 32 / 8 |
| Training epochs | 200 |
| Final training loss | ~0.06 |

---

## 🚀 Live Demo

Run the full system in **Google Colab** — no local setup required:

```python
from IPython.display import display, HTML

# Cell 1: Run this
display(HTML(open("hindi_ambiguity_hinglish.html").read()))
```

Or paste the complete HTML string directly into `display(HTML("""..."""))`.

> ⚠️ **Important:** Never paste the raw HTML file directly into a Colab cell. It will cause `SyntaxError: invalid decimal literal` because Python tries to parse CSS as Python code. Always wrap in `display(HTML("""..."""))`.

---

## ✨ Features

- 🔤 **Dual input modes** — Hindi (Devanagari script) and Hinglish (Roman/English)
- 🔍 **Fuzzy search** — finds the best match even with partial or approximate input
- 🎯 **6 ambiguity type filters** — filter by Lexical, Syntactic, Semantic, Pragmatic, Idiomatic, or Referential
- 📊 **2–4 meanings per entry** — displayed in a side-by-side 2×2 grid
- ⚡ **Instant results** — rule-based engine requires no GPU
- 🧹 **Clear button** — resets all search state in one click
- 🌐 **Zero dependencies** — single HTML file, no npm, no frameworks
- 📱 **Responsive design** — works on mobile and desktop
- 🎨 **Blue-white theme** — clean, professional interface

---

## 📚 Ambiguity Types Covered

| Type | Hindi | Example |
|------|-------|---------|
| **Lexical** | शाब्दिक | `कल` → yesterday / tomorrow |
| **Syntactic** | वाक्यरचना | `मैंने दूरबीन से आदमी को देखा` → I used binoculars / he had binoculars |
| **Semantic** | अर्थात्मक | `वह घर में घुस गया` → entered house / trespassed |
| **Pragmatic** | व्यावहारिक | `यहाँ बहुत ठंड है` → factual statement / request to close window |
| **Idiomatic** | मुहावरा | `दिल टूट गया` → heart physically broke / heartbreak |
| **Referential** | संदर्भात्मक | `मोहन ने कहा कि वह बीमार है` → Mohan is sick / Ram is sick |

---

## 📁 Project Structure

```
hindi-ambiguity-resolver/
│
├── hindi_ambiguity_hinglish.html       # Complete web interface (single file)
├── hindi_ambiguity_2B_transformer.py   # 2B parameter PyTorch model
├── Hindi_Ambiguity_Resolution_System.pptx  # Detailed presentation (15 slides)
├── README.md                           # This file
│
├── model/
│   ├── TransformerConfig               # Model hyperparameters dataclass
│   ├── RotaryEmbedding                 # RoPE positional encoding
│   ├── GroupedQueryAttention           # GQA (32Q / 8KV heads)
│   ├── SwiGLU                          # SwiGLU feed-forward block
│   ├── DecoderBlock                    # Single transformer decoder block
│   └── HindiAmbiguity2BTransformer     # Full 2B model class
│
├── data/
│   ├── HINDI_DATA                      # 30 training sentence pairs
│   └── HINGLISH_MAP                    # 100+ Hinglish → Hindi mappings
│
└── training/
    ├── HindiSPTokenizer                # SentencePiece BPE tokenizer wrapper
    ├── AmbiguityDataset                # PyTorch Dataset class
    ├── collate_fn                      # DataLoader collation with padding
    └── train()                         # Full training loop
```

---

## 🏗️ System Architecture

The system runs on two parallel tracks:

```
User Input (Hindi or Hinglish)
         │
         ▼
┌─────────────────────┐
│  Hinglish Resolver  │  ← resolveHinglish() converts Roman to Devanagari
└─────────┬───────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌────────┐  ┌──────────────┐
│ Track A│  │   Track B    │
│ Rule-  │  │  2B Param    │
│ Based  │  │ Transformer  │
│ Engine │  │    Model     │
└────┬───┘  └──────┬───────┘
     │              │
     └──────┬───────┘
            ▼
   ┌─────────────────┐
   │  Two meanings   │
   │  displayed in   │
   │  blue-white UI  │
   └─────────────────┘
```

### Track A — Rule-Based Engine

1. User inputs Hindi or Hinglish text
2. Hinglish → Hindi mapping (100+ aliases)
3. Exact key match in 60-entry database
4. Fuzzy word-overlap scoring if no exact match
5. Best-match entry selected
6. 2–4 meanings returned instantly

### Track B — 2B Transformer Model

1. Sentence tokenized via SentencePiece BPE (32k vocab)
2. Token IDs embedded (d_model = 2048)
3. RoPE positional encoding injected into Q/K matrices
4. 24 × Decoder blocks (GQA + SwiGLU + RMSNorm)
5. `<INT1>` → Interpretation 1 generated autoregressively
6. `<SEP>` → `<INT2>` → Interpretation 2 generated
7. Nucleus sampling (top-p = 0.9, temperature = 0.7)

---

## 🤖 2B Parameter Transformer Model

### Architecture Specifications

```python
TransformerConfig(
    vocab_size   = 32000,   # SentencePiece BPE vocabulary
    d_model      = 2048,    # Hidden / embedding dimension
    n_heads      = 32,      # Multi-head attention heads
    n_layers     = 24,      # Transformer decoder layers
    d_ff         = 8192,    # Feed-forward inner dimension (4 × d_model)
    max_seq_len  = 512,     # Maximum sequence length
    dropout      = 0.1,     # Dropout probability
)
```

### Components

#### 1. Token Embedding
- 32,000-token Hindi SentencePiece BPE vocabulary
- Projected into 2048-dimensional space
- Weights tied with output LM head (saves ~65M parameters)

#### 2. Rotary Position Embedding (RoPE)
```python
class RotaryEmbedding(nn.Module):
    # Injects position via rotation in Q and K matrices
    # 0 learned parameters — purely mathematical
    # Used in: LLaMA, Mistral, GPT-NeoX
```

#### 3. Grouped Query Attention (GQA)
```python
class GroupedQueryAttention(nn.Module):
    n_heads    = 32   # Query heads
    n_kv_heads = 8    # Key-Value heads (4× memory savings)
    head_dim   = 64   # d_model / n_heads
```

#### 4. SwiGLU Feed-Forward
```python
class SwiGLU(nn.Module):
    # Gate = SiLU(W1 · x) × W3 · x
    # Output = W2 · Gate
    # No bias — used in PaLM, LLaMA, Gemma
```

#### 5. RMSNorm (Pre-Norm)
- Applied **before** each sub-layer (pre-norm)
- More training-stable than post-norm
- `ε = 1e-5` for numerical stability

#### 6. Dual Output Format

The model is trained to generate sequences in this format:

```
<SOS> [source tokens] <INT1> [interpretation 1] <SEP> <INT2> [interpretation 2] <EOS>
```

Special token IDs:

| Token | ID | Purpose |
|-------|----|---------|
| `<PAD>` | 0 | Padding |
| `<SOS>` | 1 | Start of sequence |
| `<EOS>` | 2 | End of sequence |
| `<UNK>` | 3 | Unknown token |
| `<SEP>` | 4 | Separates two interpretations |
| `<INT1>` | 5 | Marks start of Interpretation 1 |
| `<INT2>` | 6 | Marks start of Interpretation 2 |

---

## 📖 Rule-Based Database

Each of the 60 database entries follows this structure:

```javascript
{
  key:      "सोना",           // Hindi word or sentence
  type:     "lexical",        // Ambiguity type
  meanings: [
    { hi: "नींद लेना",   en: "To sleep — the act of resting" },
    { hi: "स्वर्ण धातु", en: "Gold — the precious yellow metal" }
  ],
  example: "उसे सोना पसंद है | Hinglish: sona, gold"
}
```

### Distribution by Type

| Type | Count |
|------|-------|
| Lexical | 27 |
| Idiomatic | 11 |
| Pragmatic | 7 |
| Referential | 6 |
| Semantic | 6 |
| Syntactic | 5 |
| **Total** | **60** |

### Fuzzy Search Algorithm

```
Step 1 → Exact key match (O(1) JS object lookup)
Step 2 → Split input into word set qw
Step 3 → For each DB entry, split key into word set kw
Step 4 → overlap = |kw ∩ qw| (count shared words)
Step 5 → Filter entries with overlap > 0
Step 6 → Sort descending by overlap score
Step 7 → Return top matched entries
```

---

## 🔡 Hinglish Input Support

The system accepts **Romanized Hindi** as typed by 500M+ Indians daily.

### How It Works

```javascript
function resolveHinglish(q) {
  const lower = q.toLowerCase().trim();

  // Step 1: Exact key match
  if (HINGLISH_MAP[lower]) return HINGLISH_MAP[lower];

  // Step 2: Partial / substring match
  for (const [hgl, hindi] of Object.entries(HINGLISH_MAP)) {
    if (lower.includes(hgl) || hgl.includes(lower)) return hindi;
  }

  return null; // use original input for DB search
}
```

### Sample Mappings

| Hinglish Input | Hindi Output | Type |
|----------------|--------------|------|
| `kal` | कल | Transliteration |
| `sona` | सोना | Transliteration |
| `gold` | सोना | English synonym |
| `dil toot gaya` | दिल टूट गया | Phonetic sentence |
| `heartbreak` | दिल टूट गया | English meaning |
| `sitting idle` | हाथ पर हाथ धरे बैठा है | English meaning of idiom |
| `betrayal` | पीठ में छुरा घोंपना | Concept mapping |
| `fell in love` | आँखें चार हो गईं | Idiom meaning |
| `bank gaya` | वह बैंक गया | Partial sentence |
| `bahut acha` | बहुत अच्छा! | Phonetic |

---

## 🎨 Web Interface

The interface is a **single self-contained HTML file** with zero external dependencies.

### Technology Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| Structure | HTML5 | Semantic containers, accessible markup |
| Styling | CSS3 | Grid layout, CSS variables, animations |
| Logic | Vanilla JS (ES6+) | DB lookup, fuzzy match, DOM manipulation |

### CSS Color Palette

```css
--navy:    #0A2463   /* Primary background and headers */
--blue2:   #1565C0   /* Secondary elements */
--blue3:   #2979FF   /* Accent, active states */
--ice:     #E3F2FD   /* Light backgrounds */
--offwhite:#F5F9FF   /* Page background */
```

### JavaScript Functions

| Function | Purpose |
|----------|---------|
| `setMode(mode)` | Switches between Hindi and Hinglish input modes |
| `resolveHinglish(q)` | Converts Roman/English input to Devanagari |
| `buildFilters()` | Creates type filter chip buttons in the DOM |
| `setFilter(type, el)` | Applies a type filter and refreshes display |
| `buildChips()` | Creates mode-specific quick example buttons |
| `doSearch()` | Reads input, resolves Hinglish, triggers render |
| `clearAll()` | Resets search, filter, and input to defaults |
| `findBest(q)` | Exact then fuzzy matching against DB entries |
| `renderCard(e)` | Converts a DB entry into an HTML result card |
| `render()` | Main display function — search results or browse grid |
| `sel(key)` | Selects a browse card and shows its detail view |

---

## ⚙️ Installation & Setup

### Prerequisites

```bash
Python 3.10+
PyTorch 2.0+
```

### Install Dependencies

```bash
pip install torch sentencepiece accelerate
pip install bitsandbytes   # optional — for 4-bit quantization on GPU
```

### Clone and Run

```bash
git clone https://github.com/yourusername/hindi-ambiguity-resolver.git
cd hindi-ambiguity-resolver

# Run training + inference
python hindi_ambiguity_2B_transformer.py

# Open the web interface
open hindi_ambiguity_hinglish.html
```

---

## 🔬 Running in Google Colab

### Step 1 — Import

```python
from IPython.display import display, HTML
```

### Step 2 — Load the Web Interface

```python
# Option A: If file is uploaded to Colab
with open("hindi_ambiguity_hinglish.html", "r", encoding="utf-8") as f:
    display(HTML(f.read()))
```

```python
# Option B: Paste full HTML as string
display(HTML("""
<!DOCTYPE html>
... full HTML content ...
"""))
```

### Step 3 — Run the Python Backend (Optional)

```python
# Terminal output for tallying expected results
AMBIGUITY_DB = { ... }  # 60 Python entries

def resolve_terminal(sentence):
    result, matched_key = find_ambiguity(sentence)
    print(f"वाक्य     : {matched_key}")
    print(f"व्याख्या १ : {result['meanings'][0][0]}")
    print(f"व्याख्या २ : {result['meanings'][1][0]}")

resolve_terminal("मुझे सोना चाहिए")
```

### Step 4 — Train the Transformer

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
config = TransformerConfig(d_model=256, n_heads=8, n_layers=4, d_ff=512)
model  = HindiAmbiguity2BTransformer(config, n_kv_heads=4).to(device)
model  = train(model, loader, config, epochs=200, lr=5e-3, device=device)
```

---

## 📊 Usage

### Hindi Mode

Type in Devanagari script directly:

```
Input:    मुझे सोना चाहिए
Output 1: मुझे नींद लेनी है      (I need to sleep)
Output 2: मुझे स्वर्ण धातु चाहिए (I want gold)
```

### Hinglish Mode

Type in Roman script:

```
Input:    dil toot gaya
Resolved: दिल टूट गया
Output 1: हृदय को शारीरिक क्षति हुई   (Heart physically broke)
Output 2: भावनात्मक रूप से गहरा दुख हुआ (Deep emotional heartbreak)
```

### English Keywords

```
Input:    heartbreak    →  दिल टूट गया
Input:    betrayal      →  पीठ में छुरा घोंपना
Input:    sitting idle  →  हाथ पर हाथ धरे बैठा है
Input:    fell in love  →  आँखें चार हो गईं
```

---

## ✅ Expected Output

### Terminal (Python Backend)

```
Device: cpu
Running TINY config (CPU)
Vocabulary built: 312 tokens
Dataset: 30 samples, avg length: 24 tokens
Model initialized: 28.4M parameters

Epoch   1/200 | loss=4.2831 | lr=5.00e-03
Epoch  50/200 | loss=1.3872 | lr=3.45e-03
Epoch 100/200 | loss=0.5233 | lr=1.84e-03
Epoch 150/200 | loss=0.1974 | lr=6.70e-04
Epoch 200/200 | loss=0.0613 | lr=5.00e-05
Training complete. Best loss: 0.0524

वाक्य       : मुझे सोना चाहिए
┌ व्याख्या १ : मुझे नींद लेनी है
└ व्याख्या २ : मुझे स्वर्ण धातु चाहिए

वाक्य       : वह बैंक गया
┌ व्याख्या १ : वह वित्तीय संस्थान में गया
└ व्याख्या २ : वह नदी के किनारे गया
```

---

## 🐛 Common Errors & Fixes

### SyntaxError: invalid decimal literal

```
File "...", line 13
    min-height: 100vh;
                   ^
SyntaxError: invalid decimal literal
```

**Cause:** Raw HTML pasted directly into a Colab cell — Python tries to parse CSS as Python code.

**Fix:** Always wrap HTML in `display(HTML("""..."""))`:

```python
from IPython.display import display, HTML
display(HTML("""<!DOCTYPE html>...(full HTML)...</html>"""))
```

---

### `<UNK>` tokens in output

**Cause:** Tokenizer mapped unseen words to `<UNK>` (ID 3).

**Fix:** Modified `encode()` to auto-add new words to vocabulary:

```python
def encode(self, text):
    tokens = re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+|\S', text)
    for t in tokens:
        self._add_word(t)   # ← auto-expand vocab
    ids = [self.word2id[t] for t in tokens]
    return [self.sos_id] + ids + [self.eos_id]
```

---

### Session crashed — Out of RAM

**Cause:** 2B model instantiated on CPU, allocating ~8GB RAM instantly.

**Fix:** Use meta device initialization:

```python
with torch.device("meta"):
    model = HindiAmbiguity2BTransformer(config, n_kv_heads=8)
model = model.to_empty(device=device)
model.apply(model._init_weights)
```

---

### Random / gibberish output

**Cause:** Learning rate `2e-4` too slow for 30-sentence dataset.

**Fix:** Use `lr=5e-3` with `OneCycleLR` scheduler and 200 epochs:

```python
scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=5e-3,
    steps_per_epoch=len(loader),
    epochs=200, pct_start=0.1
)
```

---

### vocab_size mismatch

**Cause:** `config.vocab_size` set before `tokenizer.fit()`.

**Fix:** Set vocab size **after** fitting:

```python
tokenizer.fit(all_texts)
config.vocab_size = tokenizer.vocab_size   # ← after fit
```

---

## 🛠️ Technologies Used

### Python / ML

| Tool | Usage |
|------|-------|
| `Python 3.10+` | Base language |
| `torch` (PyTorch) | Model definition, training, inference |
| `torch.nn` | Embedding, Linear, RMSNorm, Dropout |
| `torch.optim` | AdamW + OneCycleLR scheduler |
| `torch.cuda.amp` | BF16 mixed-precision training |
| `torch.utils.data` | Dataset, DataLoader, pad_sequence |
| `sentencepiece` | BPE tokenizer for Hindi (32k vocab) |
| `bitsandbytes` | 4-bit / 8-bit quantization (optional) |
| `gc` | Garbage collection between phases |

### Web Interface

| Tool | Usage |
|------|-------|
| `HTML5` | Single-file app, no framework needed |
| `CSS3 Grid` | 2×2 meanings grid, auto-fill browse grid |
| `CSS Variables` | Centralized color palette |
| `Vanilla JS ES6+` | DB lookup, fuzzy match, DOM manipulation |
| `CSS @keyframes` | fadeIn animation on result cards |
| `CSS @media` | Responsive layout at 480px breakpoint |

### Development

| Tool | Usage |
|------|-------|
| `Google Colab` | Cloud GPU environment for training |
| `IPython.display` | `display(HTML())` for inline rendering |
| `PptxGenJS` | Programmatic slide generation |

---

## ⚡ RAM & Performance Optimization

| Technique | RAM Saving | Code |
|-----------|-----------|------|
| Meta device init | ~8GB CPU RAM | `torch.device("meta")` |
| 4-bit quantization | ~75% model size | `load_in_4bit=True` |
| Gradient checkpointing | ~60% activation RAM | `layer.gradient_checkpointing = True` |
| Gradient accumulation | Enables larger effective batch | `accum_steps=16` |
| No multiprocess loading | Avoids RAM duplication | `num_workers=0` |
| CUDA cache clearing | Prevents memory buildup | `torch.cuda.empty_cache()` |
| Tied embeddings | Saves ~65M params | `lm_head.weight = embed.weight` |

### Auto-Config by Hardware

```python
if gpu_mem_gb >= 20:
    config = TransformerConfig()           # Full 2B — needs 24GB GPU
elif gpu_mem_gb >= 10:
    config = TransformerConfig(            # ~400M — needs 12-16GB GPU
        d_model=1024, n_layers=16)
else:
    config = TransformerConfig(            # ~30M — any GPU
        d_model=512, n_layers=8)
```

---

## 🔮 Future Work

- [ ] Scale to **7B parameters** using LoRA fine-tuning on IndicBERT
- [ ] Expand database to **500+ entries** with crowd-sourced Hindi corpora
- [ ] Integrate **IndicNLP tokenizer** for superior Hindi subword handling
- [ ] Add **context window** (surrounding sentences) for pragmatic disambiguation
- [ ] **BLEU score & human evaluation** against gold-standard interpretations
- [ ] **Speech input** — ASR → ambiguity resolver pipeline for voice interfaces
- [ ] **REST API deployment** with FastAPI + streaming response support
- [ ] **Mobile-first PWA** interface with offline Hinglish-to-Hindi mapping
- [ ] **Sarcasm detection** layer for pragmatic type disambiguation
- [ ] **Multi-sentence context** — resolve pronoun references across paragraphs

---

## 📄 License

This project is released under the **MIT License**.

```
MIT License

Copyright (c) 2026 Hindi Ambiguity Resolution System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```


