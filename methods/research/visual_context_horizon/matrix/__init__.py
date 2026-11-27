"""8x A100 multimodal diffusion break-even matrix runner.

Modules are CPU-importable.  vLLM, torch and transformers are imported lazily
inside the functions that need a GPU so that planning, resume, packing and the
test suite work on a laptop.
"""
