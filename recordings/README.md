# Recorded Gemini responses

Created by a live run with `python run_pipeline.py --provider gemini --record`.
Commit the `*.json` files here so anyone can run `python run_pipeline.py --provider replay`
and see the real Gemini extraction and adjudication without an API key or network access.
Re-record after changing prompts, the model or the input documents (the cache key covers all three).
