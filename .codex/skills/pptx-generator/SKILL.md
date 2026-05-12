---
name: pptx-generator
description: "Generate, edit, and read PowerPoint presentations. Create PPTX files directly with the bundled python-pptx generator, create highly custom decks with PptxGenJS, edit existing PPTX files via XML workflows, or extract presentation text for QA. Triggers: PPT, PPTX, PowerPoint, presentation, slide, deck, slides."
license: MIT
metadata:
  version: "1.1"
  category: productivity
  sources:
    - https://python-pptx.readthedocs.io/
    - https://gitbrent.github.io/PptxGenJS/
    - https://github.com/microsoft/markitdown
---

# PPTX Generator & Editor

## Overview

This skill handles PowerPoint work in three modes:

1. Direct `.pptx` generation with the bundled local Python scripts.
2. Highly custom slide creation with PptxGenJS.
3. Editing existing presentations by unpacking and modifying the PPTX XML.

Use the bundled Python path first when the user wants a `.pptx` file quickly and there is no strong need for bespoke slide engineering.

## Quick Reference

| Task | Approach |
|------|----------|
| Generate a PPTX quickly | `python .codex/skills/pptx-generator/scripts/generate_pptx.py --input deck.json --output output.pptx` |
| Inspect visible slide text | `python .codex/skills/pptx-generator/scripts/inspect_pptx.py output.pptx` |
| Read/analyze content deeply | `python -m markitdown presentation.pptx` |
| Edit an existing PPTX | See [Editing Presentations](references/editing.md) |
| Create a highly custom deck | Use the PptxGenJS workflow below |

## Bundled Local Workflow

### Included files

| File | Purpose |
|------|---------|
| [scripts/generate_pptx.py](scripts/generate_pptx.py) | Build a deck from structured JSON with `python-pptx` |
| [scripts/inspect_pptx.py](scripts/inspect_pptx.py) | Extract visible text for quick QA |
| [assets/sample_deck.json](assets/sample_deck.json) | Example input schema |

### Use this mode when

- The user explicitly wants a `.pptx` file now.
- The environment already has `python-pptx`.
- A clean editable deck matters more than custom one-off layout code.

### Fast path

1. Convert the request into deck JSON.
2. Run the bundled generator.
3. Inspect the visible text before declaring success.

Example:

```bash
python .codex/skills/pptx-generator/scripts/generate_pptx.py \
  --input .codex/skills/pptx-generator/assets/sample_deck.json \
  --output artifacts/sample-presentation.pptx
```

```bash
python .codex/skills/pptx-generator/scripts/inspect_pptx.py artifacts/sample-presentation.pptx
```

### Deck JSON contract

Top-level keys:

- `title`: deck title
- `subtitle`: optional subtitle
- `author`: optional footer on the title slide
- `theme`: optional theme override
- `slides`: ordered slide list

Theme keys:

- `primary`
- `secondary`
- `accent`
- `light`
- `bg`
- `font_cn`
- `font_en`

Supported slide types:

- `title`
- `section`
- `bullets`
- `two_column`
- `summary`

Minimal `bullets` slide example:

```json
{
  "type": "bullets",
  "title": "System Overview",
  "lead": "One framing sentence.",
  "bullets": [
    "Point one",
    "Point two",
    "Point three"
  ],
  "highlight": "Optional badge"
}
```

### Default visual contract

- Deck size: 10" x 5.625" (`16:9`)
- Default English font: Arial
- Default Chinese font: Microsoft YaHei
- Every slide except the title slide gets a page number badge
- Colors must be 6-char hex; `#` is tolerated by the Python generator but should still be avoided for consistency with the PptxGenJS path

## Reference Files

| File | Contents |
|------|----------|
| [slide-types.md](references/slide-types.md) | Slide categories and layout patterns |
| [design-system.md](references/design-system.md) | Color palettes, font guidance, and style recipes |
| [editing.md](references/editing.md) | Template-based editing workflow and XML guidance |
| [pitfalls.md](references/pitfalls.md) | QA process and common mistakes |
| [pptxgenjs.md](references/pptxgenjs.md) | PptxGenJS API reference |

## Reading Content

Preferred when available:

```bash
python -m markitdown presentation.pptx
```

Fallback:

```bash
python .codex/skills/pptx-generator/scripts/inspect_pptx.py presentation.pptx
```

Use `markitdown` when you need richer extraction. Use the bundled inspector when you just need visible slide text for a QA pass.

## Direct Generation Workflow

Use this default execution path unless the user needs a pre-existing template or unusually custom layouts.

1. Determine the audience, purpose, tone, and slide count.
2. Draft slide content in structured JSON.
3. Choose a theme from [design-system.md](references/design-system.md) if the user cares about styling.
4. Generate the deck with `generate_pptx.py`.
5. Run `inspect_pptx.py`.
6. Fix content issues and regenerate if needed.

If the user wants diagrams, generate them separately and then either:

- Use the PptxGenJS path for image-rich custom placement, or
- Extend the Python generator for the specific image placement needed.

## Creating from Scratch with PptxGenJS

Use this mode when no template is provided and the deck needs stronger visual customization than the bundled Python generator supports.

### Workflow

1. Research the user requirements: topic, audience, purpose, tone, and content depth.
2. Select a color palette and font pairing from [design-system.md](references/design-system.md).
3. Choose a visual style recipe.
4. Plan the slide outline and classify each slide using [slide-types.md](references/slide-types.md).
5. Create one JS file per slide in `slides/`.
6. Create `slides/compile.js` to combine the slide modules.
7. Run the compile script and perform QA.

### Required PptxGenJS rules

- Each slide file must export a synchronous `createSlide(pres, theme)` function.
- Do not use `async` or `await` in `createSlide`.
- Do not use `#` in hex colors.
- Do not reuse mutated option objects across calls.
- Add page number badges to all slides except the cover slide.

### Compile example

```javascript
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";

const theme = {
  primary: "22223b",
  secondary: "4a4e69",
  accent: "9a8c98",
  light: "c9ada7",
  bg: "f2e9e4"
};

for (let i = 1; i <= 12; i++) {
  const num = String(i).padStart(2, "0");
  const slideModule = require(`./slide-${num}.js`);
  slideModule.createSlide(pres, theme);
}

pres.writeFile({ fileName: "./output/presentation.pptx" });
```

## Editing Existing Presentations

For template-based deck editing, follow [editing.md](references/editing.md).

Use that workflow when:

- The user provides an existing `.pptx` template.
- Slide masters, placements, or house style must be preserved exactly.
- You need to replace content without rebuilding the deck from scratch.

## QA

Always assume the first render has issues.

Minimum QA loop:

1. Generate the deck.
2. Extract visible text with `inspect_pptx.py` or `markitdown`.
3. Check for missing text, wrong order, leftover placeholders, and malformed headings.
4. Fix and regenerate.
5. Re-run QA before finishing.

See [pitfalls.md](references/pitfalls.md) for the stricter checklist.

## Dependencies

- `python-pptx` for the bundled generator and inspector
- `pip install "markitdown[pptx]"` for richer extraction
- `npm install -g pptxgenjs` for advanced custom deck creation
- `npm install -g react-icons react react-dom sharp` for optional icon workflows
