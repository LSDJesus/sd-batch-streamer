#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.5.8
#
# Changelog:
# v0.5.8: Final Stability Fix. Correctly implemented the dictionary-based output model for
#         all event handlers, resolving the "didn't receive enough output values" ValueError.
# v0.5.7: (Failed Attempt) Incorrectly configured event handlers.
# v0.5.6: Compatibility Fix. Removed reference to `shared.sd_vae`.
# v0.5.5: Critical Bug Fix. Corrected Python scope error.
#

import gradio as gr
import itertools
import functools
from PIL import Image, ImageDraw, ImageFont
from modules import shared, processing, sd_samplers, sd_schedulers
from modules.script_callbacks import on_ui_tabs

# --- Version Information ---
__version__ = "0.5.8"

# --- Globals to store data between button clicks ---
last_run_images = []
last_run_labels = {}

# --- Helper Functions ---
def create_labeled_grid(images, x_labels, y_labels, font_size=30, margin=5):
    if not all([images, x_labels, y_labels]): return None
    grid_cols, grid_rows = len(x_labels), len(y_labels)
    if len(images) != grid_cols * grid_rows: return None
    img_w, img_h = images[0].size
    label_area = font_size + 2 * margin
    grid_w, grid_h = img_w * grid_cols + label_area, img_h * grid_rows + label_area
    grid_image = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid_image)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except IOError: font = ImageFont.load_default()
    for i, label in enumerate(y_labels):
        draw.text((margin, label_area + i * img_h + (img_h - font_size) // 2), str(label), fill="black", font=font)
    for i, label in enumerate(x_labels):
        text_bbox = draw.textbbox((0, 0), str(label), font=font)
        text_w = text_bbox[2] - text_bbox[0]
        draw.text((label_area + i * img_w + (img_w - text_w) // 2, margin), str(label), fill="black", font=font)
    for i, img in enumerate(images):
        grid_image.paste(img, (label_area + (i % grid_cols) * img_w, label_area + (i // grid_cols) * img_h))
    return grid_image

# --- Main Logic Functions ---

def run_xyz_matrix(
    components, # A dictionary of the UI components we need to update
    p_prompt, n_prompt, sampler, scheduler, steps, cfg, width, height, seed,
    x_values_str, y_values_str, z_values_str
):
    global last_run_images, last_run_labels
    last_run_images, last_run_labels = [], {}

    yield {
        components['log']: "Parsing inputs...",
        components['gen_btn']: gr.Button.update(interactive=False),
        components['asm_btn']: gr.Button.update(visible=False)
    }
    
    x_vals = [x.strip() for x in x_values_str.split('|') if x.strip()]
    y_vals = [y.strip() for y in y_values_str.split('|') if y.strip()]
    z_vals = [z.strip() for z in z_values_str.split('|') if z.strip()]
    total_images = len(x_vals) * len(y_vals) * len(z_vals)
    if total_images == 0:
        yield { components['log']: "Error: No jobs to run.", components['gen_btn']: gr.Button.update(interactive=True) }; return
    
    job_list = list(itertools.product(z_vals, y_vals, x_vals))
    all_images = []
    
    yield { components['log']: f"Starting generation of {total_images} images...", components['ind_gallery']: None }
    shared.state.begin(); shared.state.job_count = total_images
    for i, (z_val, y_val, x_val) in enumerate(job_list):
        if shared.state.interrupted: break
        shared.state.job = f"Image {i+1}/{total_images}"
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model, prompt=p_prompt.replace("{subject}", x_val), negative_prompt=n_prompt,
            steps=int(steps), cfg_scale=float(y_val), sampler_name=z_val, scheduler=scheduler,
            width=int(width), height=int(height), seed=int(seed) + i, n_iter=1, batch_size=1, do_not_save_samples=True,
        )
        proc = processing.process_images(p)
        if proc.images:
            all_images.append(proc.images[0])
            yield { components['ind_gallery']: all_images, components['log']: f"Generated image {i+1}/{total_images}" }
    shared.state.end()

    last_run_images = all_images
    last_run_labels = {'x': x_vals, 'y': y_vals, 'z': z_vals}
    
    yield {
        components['log']: f"Generation of {len(all_images)} images complete. Ready to assemble grids.",
        components['gen_btn']: gr.Button.update(interactive=True),
        components['asm_btn']: gr.Button.update(visible=True)
    }

def assemble_grids_from_last_run(components):
    if not last_run_images or not last_run_labels:
        yield { components['log']: "No images from a previous run found to assemble." }; return

    x_vals, y_vals, z_vals = last_run_labels['x'], last_run_labels['y'], last_run_labels['z']
    
    yield { components['log']: "Assembling X/Y grids..." }
    grid_images = []
    images_per_grid = len(x_vals) * len(y_vals)
    for i, z_label in enumerate(z_vals):
        grid_data = last_run_images[i * images_per_grid:(i + 1) * images_per_grid]
        grid = create_labeled_grid(grid_data, x_vals, y_vals)
        if grid: grid_images.append(grid)
    
    if len(grid_images) > 1:
        yield { components['grid_gallery']: grid_images, components['log']: "Assembling Mega-Grid..." }
        grid_w, grid_h = grid_images[0].size
        mega_grid = Image.new('RGB', (grid_w * len(grid_images), grid_h), 'white')
        for i, grid_img in enumerate(grid_images):
            mega_grid.paste(grid_img, (i * grid_w, 0))
        yield { components['mega_grid']: mega_grid, components['log']: "All grids assembled." }
    else:
        yield { components['grid_gallery']: None, components['mega_grid']: grid_images[0] if grid_images else None, components['log']: "Grid assembled." }


def create_streamer_ui():
    with gr.Blocks() as streamer_tab:
        gr.HTML(f"<h3>XYZ Matrix POC <span style='font-size:0.8rem;color:grey;'>v{__version__}</span></h3>")
        with gr.Accordion("Global Settings", open=True):
            with gr.Row():
                sampler_dropdown = gr.Dropdown(label='Sampler', choices=[s.name for s in sd_samplers.all_samplers], value='Euler a')
                scheduler_dropdown = gr.Dropdown(label='Scheduler', choices=[s.label for s in sd_schedulers.schedulers], value='Automatic')
            with gr.Row():
                steps_slider = gr.Slider(minimum=1, maximum=150, step=1, label="Steps", value=20)
                cfg_slider = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
            with gr.Row():
                width_slider = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                height_slider = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
            seed_input = gr.Number(label="Seed", value=-1, precision=0)
        positive_prompt = gr.Textbox(label="Positive Prompt", placeholder="Use {subject} where you want the X-axis value to appear.", value="photo of a {subject}, high quality")
        negative_prompt = gr.Textbox(label="Negative Prompt", value="low quality, bad anatomy")
        with gr.Accordion("Matrix Inputs", open=True):
            x_input = gr.Textbox(label="X Axis (Subject)", placeholder="Use '|' to separate values", value="dog|woman|elephant")
            y_input = gr.Textbox(label="Y Axis (CFG Scale)", placeholder="Use '|' to separate values", value="3|6|9")
            z_input = gr.Textbox(label="Z Axis (Sampler)", placeholder="Use '|' to separate values", value="Euler a|DPM++ 2M Karras|DPM2")
        
        with gr.Row():
            generate_button = gr.Button("Generate XYZ Matrix", variant="primary")
            assemble_button = gr.Button("Assemble Grids from Last Run", variant="secondary", visible=False)
            
        html_log = gr.HTML(label="Log")
        with gr.Tabs():
            with gr.TabItem("Live Individual Images"):
                individual_gallery = gr.Gallery(label="Individual Generations", show_label=False, columns=9)
            with gr.TabItem("X/Y Grids"):
                grid_gallery = gr.Gallery(label="X/Y Grids per Z-Value", show_label=False, columns=3)
            with gr.TabItem("Mega-Grid"):
                mega_grid_image = gr.Image(label="Final Mega-Grid", show_label=False)

        # --- Event Handlers (FIXED) ---
        
        # A dictionary to pass component objects to the handler functions
        gen_components = {
            'log': html_log, 'gen_btn': generate_button, 'asm_btn': assemble_button,
            'ind_gallery': individual_gallery
        }
        gen_inputs = [
            positive_prompt, negative_prompt, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, width_slider, height_slider, seed_input,
            x_input, y_input, z_input
        ]
        # Create a function call that includes the components dictionary as the first argument
        fn_with_components = functools.partial(run_xyz_matrix, gen_components)
        # The outputs parameter MUST be None when the function yields dictionaries
        generate_button.click(fn=fn_with_components, inputs=gen_inputs, outputs=None)
        
        # Do the same for the assemble button
        asm_components = {
            'log': html_log, 'grid_gallery': grid_gallery, 'mega_grid': mega_grid_image
        }
        asm_fn_with_components = functools.partial(assemble_grids_from_last_run, asm_components)
        assemble_button.click(fn=asm_fn_with_components, inputs=None, outputs=None)

    return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "XYZ Matrix POC", "xyz_matrix_poc_tab")]

on_ui_tabs(add_new_tab_to_ui)