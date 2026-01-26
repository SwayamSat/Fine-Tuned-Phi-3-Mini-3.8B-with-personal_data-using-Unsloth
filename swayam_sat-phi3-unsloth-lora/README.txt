PLEASE READ:

This folder `phi3-unsloth-lora` is where you must place your fine-tuned adapter files.
You mentioned your adapter directory is named `swayam_sat-phi3-unsloth-lora`.

Please COPY the contents of `swayam_sat-phi3-unsloth-lora` into this folder, OR rename your folder to `phi3-unsloth-lora` and replace this one.

The expected files are:
- adapter_model.safetensors
- adapter_config.json
- tokenizer.json (and other tokenizer files)

This path matches the `ADAPTER_PATH = "phi3-unsloth-lora"` configuration in `app.py`.
