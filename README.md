# mojo-einsum

Mixed Python and Mojo package scaffolded by mohaus.

```bash
nix develop
nix develop -c pre-commit run --all-files
python -c "import mojo_einsum; print(mojo_einsum.passthrough('hello'))"
```

Without Nix:

```bash
uv pip install -e .
python -c "import mojo_einsum; print(mojo_einsum.passthrough('hello'))"
```
