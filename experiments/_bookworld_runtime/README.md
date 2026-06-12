# BookWorld-runtime variants (reference only)

Scripts here drive experiments through the **BookWorld simulation runtime**
(`Server` → `Performer.single_role_interact`). That runtime — the product
`server.py` and the full `Performer` integration — is **not shipped** in this
repository; it belongs to the upstream [BookWorld](https://github.com/alienet1109/BookWorld)
project on which PersonaForge was built.

They are kept only to document how PersonaForge plugged into a multi-agent
world. **They will not run as-is** (they raise a clear `ImportError` explaining
the missing dependency).

For a self-contained, dependency-free long-dialogue benchmark that needs only an
API key, use [`../sft/long_dialogue_4way.py`](../sft/long_dialogue_4way.py).
