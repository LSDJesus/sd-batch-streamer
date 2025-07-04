#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.4.1
#
# Changelog:
# v0.4.1: Bug fix. Corrected Gradio event handler configuration by removing redundant 'outputs'
#         argument when a dictionary is returned, fixing a UI loading crash.
# v0.4.0: Major Feature Update. Added params.txt persistence, preset management, and apply button.
# v0.3.1: Bug fix. Explicitly set do_not_save_samples=True.
#

import gradio as gr
import json
import os
import re
import tempfile
from modules import shared, processing, sd_samplers
from modules.script_callbacks import on_ui_tabs
from modules.paths_internal import script_path
from modules.ui_components import ToolButton

# --- Version Information ---
__version__ = "0.4.1"

# --- Globals for this script ---
image_params_storage = {}
params_file_path = os.path.join(script_path, "params.txt")

# --- Helper Functions ---

def parse_infotext(infotext):
    """Parses a string of infotext and returns a dictionary of parameters."""
    params = {}
    if not infotext: return params
    
    prompt_text = infotext.split("Negative prompt:")[0].strip()
    params['prompt'] = prompt_text
    
    neg_prompt_match = re.search(r"Negative prompt: (.*?)\nSteps:", infotext, re.DOTALL)
    if neg_prompt_match:
        params['negative_prompt'] = neg_prompt_match.group(1).strip()
    
    patterns = {
        'steps': r"Steps: (\d+)", 'sampler_name': r"Sampler: ([\w\+\s]+),?",
        'cfg_scale': r"CFG scale: ([\d\.]+)", 'seed': r"Seed: (\d+)",
        'width': r"Size: (\d+)x\d+", 'height': r"Size: \d+x(\d+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, infotext)
        if match:
            value = match.group(1)
            if key in ['steps', 'seed', 'width', 'height']: params[key] = int(value)
            elif key == 'cfg_scale': params[key] = float(value)
            else: params[key] = value.strip()
                
    return params

def create_infotext(params):
    """Creates an infotext string from a dictionary of parameters."""
    p = {
        'prompt': params.get('prompt', ''), 'negative_prompt': params.get('negative_prompt', ''),
        'steps': params.get('steps', 20), 'sampler_name': params.get('sampler_name', 'Euler a'),
        'cfg_scale': params.get('cfg_scale', 7.0), 'seed': params.get('seed', -1),
        'width': params.get('width', 512), 'height': params.get('height', 512),
    }
    infotext = f"{p['prompt']}\nNegative prompt: {p['negative_prompt']}\n"
    infotext += f"Steps: {p['steps']}, Sampler: {p['sampler_name']}, CFG scale: {p['cfg_scale']}, Seed: {p['seed']}, Size: {p['width']}x{p['height']}"
    return infotext

def create_streamer_ui():
    """Creates the Gradio UI components for our tab."""
    default_params = {
        'prompt': '', 'negative_prompt': '', 'steps': 20, 'cfg_scale': 7.0,
        'width': 512, 'height': 512, 'sampler_name': 'Euler a'
    }
    if os.path.exists(params_file_path):
        with open(params_file_path, 'r', encoding='utf-8') as f:
            try:
                loaded_params = parse_infotext(f.read())
                default_params.update(loaded_params)
            except Exception as e:
                print(f"SD Batch Streamer: Could not parse params.txt, using defaults. Error: {e}")

    with gr.Blocks() as streamer_tab:
        
        def process_and_stream_images(prompts_text, negative_prompt, steps_val, cfg_val, width, height, sampler_name):
            first_prompt = prompts_text.splitlines()[0].strip() if prompts_text else ""
            current_params = {
                'prompt': first_prompt, 'negative_prompt': negative_prompt, 'steps': steps_val,
                'cfg_scale': cfg_val, 'width': width, 'height': height, 'sampler_name': sampler_name
            }
            try:
                with open(params_file_path, 'w', encoding='utf-8') as f:
                    f.write(create_infotext(current_params))
            except Exception as e:
                print(f"SD Batch Streamer: Error writing to params.txt: {e}")
            
            global image_params_storage; image_params_storage.clear()
            yield {output_gallery: gr.Gallery.update(value=[], visible=True)}
            prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
            if not prompts: yield {output_gallery: []}; return
            print(f"SD Batch Streamer (v{__version__}): Starting generation for {len(prompts)} prompts.")
            all_images = []
            yield {stream_button: gr.Button.update(value="Generating...", interactive=False)}
            shared.state.begin(); shared.state.job_count = len(prompts)
            for i, prompt in enumerate(prompts):
                shared.state.job = f"Prompt: {prompt[:80]}..."; shared.state.job_no = i + 1
                if shared.state.interrupted: break
                p = processing.StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model, prompt=prompt, negative_prompt=negative_prompt, steps=int(steps_val),
                    cfg_scale=float(cfg_val), sampler_name=sampler_name, seed=-1, width=int(width), height=int(height),
                    n_iter=1, batch_size=1, do_not_save_samples=True,
                )
                processed = processing.process_images(p)
                if processed.images:
                    new_image = processed.images[0]
                    all_images.append(new_image)
                    image_params_storage[i] = {
                        "prompt": prompt, "negative_prompt": negative_prompt, "steps": steps_val, "cfg_scale": cfg_val,
                        "width": width, "height": height, "sampler_name": sampler_name, "seed": processed.seed
                    }
                    yield {output_gallery: all_images}
            shared.state.end()
            yield {
                output_gallery: all_images,
                stream_button: gr.Button.update(value="Generate and Stream", interactive=True)
            }

        def on_gallery_select(evt: gr.SelectData):
            params = image_params_storage.get(evt.index)
            if not params: return {}
            return {
                prompts_input: f"{params['prompt']}\n", negative_prompt: params["negative_prompt"], steps: params["steps"],
                cfg_scale: params["cfg_scale"], width: params["width"], height: params["height"], sampler: params["sampler_name"]
            }

        def apply_parameters(prompts_text):
            text_to_parse = prompts_text.strip()
            if not text_to_parse and os.path.exists(params_file_path):
                with open(params_file_path, 'r', encoding='utf-8') as f: text_to_parse = f.read()
            if not text_to_parse: return {}
            params = parse_infotext(text_to_parse)
            return {
                prompts_input: params.get('prompt', ''), negative_prompt: params.get('negative_prompt', ''),
                steps: params.get('steps', 20), cfg_scale: params.get('cfg_scale', 7.0),
                width: params.get('width', 512), height: params.get('height', 512),
                sampler: params.get('sampler_name', 'Euler a')
            }

        def save_preset(prompts, neg, steps_val, cfg, w, h, samp):
            preset = {'prompt': prompts, 'negative_prompt': neg, 'steps': steps_val, 'cfg_scale': cfg,
                      'width': w, 'height': h, 'sampler_name': samp}
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
                json.dump(preset, f, indent=4); return f.name

        def load_preset(json_file):
            if json_file is None: return {}
            with open(json_file.name, 'r', encoding='utf-8') as f: preset = json.load(f)
            return {
                prompts_input: preset.get('prompt', ''), negative_prompt: preset.get('negative_prompt', ''),
                steps: preset.get('steps', 20), cfg_scale: preset.get('cfg_scale', 7.0),
                width: preset.get('width', 512), height: preset.get('height', 512),
                sampler: preset.get('sampler_name', 'Euler a')
            }

        # --- UI Layout ---
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(f"<h3>SD Batch Streamer <span style='font-size:0.8rem;color:grey;'>v{__version__}</span></h3>")
                with gr.Row():
                    prompts_input = gr.Textbox(label="Prompts (one per line)", lines=8, placeholder="A beautiful cat...", value=default_params['prompt'])
                    apply_button = ToolButton(value="↙️", tooltip="Apply parameters from prompt box or last generation.")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=3, placeholder="ugly, deformed...", value=default_params['negative_prompt'])
                with gr.Row():
                    stream_button = gr.Button("Generate and Stream", variant="primary")
                    interrupt_button = gr.Button("Interrupt", variant="secondary")
                    skip_button = gr.Button("Skip", variant="secondary")
                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=150, step=1, label="Steps", value=default_params['steps'])
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=default_params['cfg_scale'])
                with gr.Row():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=default_params['width'])
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=default_params['height'])
                sampler_choices = [s.name for s in sd_samplers.all_samplers]
                sampler = gr.Dropdown(label='Sampling method', choices=sampler_choices, value=default_params['sampler_name'])
                with gr.Row():
                    save_preset_button = gr.Button("Save Preset")
                    load_preset_button = gr.UploadButton("Load Preset", file_types=['.json'])
                save_file_output = gr.File(label="Download Preset", visible=False)
            with gr.Column(scale=3):
                gr.HTML("<h4>Live Output</h4> <p>Click an image in the gallery to send its settings back to the controls.</p>")
                output_gallery = gr.Gallery(
                    label="Live Output", show_label=False, elem_id="sd_batch_stream_gallery",
                    columns=4, rows=2, object_fit="contain", height="auto"
                )

        # --- Event Handlers (FIXED) ---
        all_ui_components = [prompts_input, negative_prompt, steps, cfg_scale, width, height, sampler]
        
        stream_button.click(fn=process_and_stream_images, inputs=all_ui_components, outputs=[output_gallery, stream_button])
        interrupt_button.click(fn=lambda: shared.state.interrupt(), inputs=None, outputs=None)
        skip_button.click(fn=lambda: shared.state.skip(), inputs=None, outputs=None)
        
        # FIX: Removed the 'outputs' argument since the function returns a dictionary.
        apply_button.click(fn=apply_parameters, inputs=[prompts_input], outputs=None)
        output_gallery.select(fn=on_gallery_select, inputs=None, outputs=None)
        load_preset_button.upload(fn=load_preset, inputs=[load_preset_button], outputs=None)
        
        # This handler remains correct as it does not return a dictionary.
        save_preset_button.click(fn=save_preset, inputs=all_ui_components, outputs=[save_file_output])

        return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "Batch Streamer", "sd_batch_streamer_tab")]

on_ui_tabs(add_new_tab_to_ui)