# Character data — bring your own

To respect copyright, **this repository ships no character data derived from copyrighted (or otherwise restricted) works.** PersonaForge is released as *code + schemas*; you reconstruct the character profiles you want to study from sources you are allowed to use.

This directory gives you everything needed to do that:

| File | Purpose |
|------|---------|
| `role_info.schema.json` | JSON Schema for a single character persona profile. |
| `examples/role_info.example.json` | A complete, **original** example character (`Lacia·Eldridge`, not derived from any real work). |
| `examples/world_info.example.json` | Example world description. |
| `examples/locations.example.json` | Example location list. |

## Where the data goes

PersonaForge loaders expect this layout (the repo keeps one original example world under `data/` so the structure is runnable out of the box):

```
data/
└── roles/
    └── <WorldName>/
        └── <RoleCode>/          # e.g. MyHero-en
            └── role_info.json    # conforms to role_info.schema.json
```

## Building a profile (Automated Parameter Acquisition)

The three-layer profile is **not** copyrighted text — it is an *analytical derivative*: Big Five scores in `[0,1]`, a Vaillant defense mechanism, a speaking-style matrix, and an initial dynamic state. As described in the paper (§3.1 *Parameter Acquisition*), you can produce one from raw text (e.g. a public wiki page) instead of hand-authoring it:

1. Gather public-domain or self-authored material about the character (a public wiki link, your own notes, an original character you invented).
2. Prompt an LLM to extract the Big Five scores, the primary defense mechanism, and the speaking-style fields, emitting JSON that validates against `role_info.schema.json`.
3. Write a few **original** `style_examples` `(context, response)` pairs in the character's voice — paraphrase, never paste verbatim source dialogue.
4. Save to `data/roles/<World>/<RoleCode>/role_info.json` and validate.

Validate with any JSON-Schema tool, e.g.:

```bash
python -c "import json,jsonschema; jsonschema.validate(json.load(open('data/roles/MyWorld/MyHero-en/role_info.json')), json.load(open('schemas/role_info.schema.json')))"
```

## What is intentionally absent

The published experiments reference character codes (e.g. classical-literature and fantasy characters) that we evaluated in the paper. We do **not** ship those profiles. To reproduce a specific paper number, recreate the corresponding character file from public sources under its expected `role_code`, following this schema.
