# Repository Guidelines

## Project Structure & Module Organization
- `t2i_sd.py` is the main entry point for text-to-image generation and CLI usage.
- `stable_diffusion/` contains core model code: UNet, VAE, sampler, tokenizer, and model I/O utilities.
- `requirements.txt` lists Python dependencies needed to run the project.
- No dedicated `tests/` directory is present in this repository.

## Build, Test, and Development Commands
- `python3 -m venv .venv && source .venv/bin/activate`: create and activate a local virtual environment.
- `pip install -r requirements.txt`: install runtime dependencies.
- `python3 t2i_sd.py`: generate an image using defaults (writes `output.png`).
- Example with environment overrides:
  `SD_PROMPT="flowers" SD_NEGATIVE_PROMPT="humans" SD_STEPS=28 SD_SEED=56 python3 t2i_sd.py`
- Optional flags: `-o output.png` to change the filename, `--no-novelai` to use `CompVis/stable-diffusion-v1-4`.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 formatting.
- Prefer `snake_case` for variables/functions and `UpperCamelCase` for classes.
- Keep functions small and explicit; avoid adding heavy abstractions without need.
- No formatter/linter is configured; if you introduce one, document it here.

## Testing Guidelines
- No automated tests are currently defined.
- If you add tests, place them under `tests/` and name files like `test_*.py`.
- Include a minimal smoke test that imports `stable_diffusion` and runs a short (low-step) generation.

## Commit & Pull Request Guidelines
- Git history shows short, descriptive messages (e.g., “Update README.md”, “various fixes”).
- Use concise, imperative commit subjects; avoid multi-line bodies unless necessary.
- PRs should include a summary, key commands run, and any example outputs or screenshots when changing model behavior or image quality.

## Configuration & Runtime Notes
- The project downloads model assets via `huggingface_hub`; expect first-run downloads and local caching.
- Key environment variables in `t2i_sd.py`: `SD_PROMPT`, `SD_NEGATIVE_PROMPT`, `SD_CFG`, `SD_STEPS`, `SD_SEED`.
