#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.6.0
#
# Changelog:
# v0.6.0: Final Stability Fix. Implemented the strict Gradio dictionary-based output model:
#         1. The 'outputs' parameter of .click() MUST contain a list of all potential output component objects.
#         2. The yielded dictionaries MUST use these component objects as their keys.
#         3. ALL values in the yielded dictionaries MUST be wrapped in gr.Component.update() calls.
#         This resolves the "Returned component not specified as output" ValueError and other related issues.
# v0.5.9: (Failed Attempt) Incorrectly applied previous fix.
# v0.5.8: (Failed Attempt) Incorrectly removed 'outputs' parameter.
#

import gradio as gr
import itertools
import functools
from PIL import Image, ImageDraw, ImageFont
from modules import shared, processing, sd_samplers, sd_schedulers
from modules.script_callbacks import on_ui_tabs

# --- Version Information ---
__version__ = "0.6.0"

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
    ui_components_map, # A dictionary of the UI components we need to update
    p_prompt, n_prompt, sampler, scheduler, steps, cfg, width, height, seed,
    x_values_str, y_values_str, z_values_str
):
    # Unpack components from the map for easier access
    html_log = ui_components_map['html_log']
    generate_button = ui_components_map['generate_button']
    assemble_button = ui_components_map['assemble_button']
    individual_gallery = ui_components_map['individual_gallery']
    grid_gallery = ui_components_map['grid_gallery']
    mega_grid_image = ui_components_map['mega_grid_image']
    
    global last_run_images, last_run_labels
    last_run_images, last_run_labels = [], {}

    # Yield updates using component objects as keys, wrapped in gr.Component.update()
    yield {
        html_log: gr.HTML.update(value="Parsing inputs..."),
        generate_button: gr.Button.update(interactive=False),
        assemble_button: gr.Button.update(visible=False),
        individual_gallery: gr.Gallery.update(value=None), # Clear gallery
        grid_gallery: gr.Gallery.update(value=None), # Clear grids
        mega_grid_image: gr.Image.update(value=None) # Clear mega grid
    }
    
    x_vals = [x.strip() for x in x_values_str.split('|') if x.strip()]
    y_vals = [y.strip() for y in y_values_str.split('|') if y.strip()]
    z_vals = [z.strip() for z in z_values_str.split('|') if z.strip()]
    total_images = len(x_vals) * len(y_vals) * len(z_vals)
    if total_images == 0:
        yield { html_log: gr.HTML.update(value="Error: No jobs to run."), generate_button: gr.Button.update(interactive=True) }; return
    
    job_list = list(itertools.product(z_vals, y_vals, x_vals))
    all_images = []
    
    yield { html_log: gr.HTML.update(value=f"Starting generation of {total_images} images...") }
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
            yield { individual_gallery: gr.Gallery.update(value=all_images), html_log: gr.HTML.update(value=f"Generated image {i+1}/{total_images}") }
    shared.state.end()

    last_run_images = all_images
    last_run_labels = {'x': x_vals, 'y': y_vals, 'z': z_vals}
    
    yield {
        html_log: gr.HTML.update(value=f"Generation of {len(all_images)} images complete. Ready to assemble grids."),
        generate_button: gr.Button.update(interactive=True),
        assemble_button: gr.Button.update(visible=True)
    }

def assemble_grids_from_last_run(ui_components_map):
    # Unpack components
    html_log = ui_components_map['html_log']
    grid_gallery = ui_components_map['grid_gallery']
    mega_grid_image = ui_components_map['mega_grid_image']

    if not last_run_images or not last_run_labels:
        yield { html_log: gr.HTML.update(value="No images from a previous run found to assemble.") }; return

    x_vals, y_vals, z_vals = last_run_labels['x'], last_run_labels['y'], last_run_labels['z']
    
    yield { html_log: gr.HTML.update(value="Assembling X/Y grids...") }
    grid_images = []
    images_per_grid = len(x_vals) * len(y_vals)
    for i, z_label in enumerate(z_vals):
        grid_data = last_run_images[i * images_per_grid:(i + 1) * images_per_grid]
        grid = create_labeled_grid(grid_data, x_vals, y_vals)
        if grid: grid_images.append(grid)
    
    if len(grid_images) > 1:
        yield { grid_gallery: gr.Gallery.update(value=grid_images), html_log: gr.HTML.update(value="Assembling Mega-Grid...") }
        grid_w, grid_h = grid_images[0].size
        mega_grid = Image.new('RGB', (grid_w * len(grid_images), grid_h), 'white')
        for i, grid_img in enumerate(grid_images):
            mega_grid.paste(grid_img, (i * grid_w, 0))
        yield { mega_grid_image: gr.Image.update(value=mega_grid), html_log: gr.HTML.update(value="All grids assembled.") }
    else:
        yield { grid_gallery: gr.Gallery.update(value=None), mega_grid_image: gr.Image.update(value=grid_images[0] if grid_images else None), html_log: gr.HTML.update(value="Grid assembled.") }


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

        # --- Event Handlers (FINAL, DEFINITIVE FIX) ---
        
        # This dictionary holds references to the UI components.
        # It's passed to the logic functions via functools.partial.
        all_ui_components_map = {
            'html_log': html_log,
            'generate_button': generate_button,
            'assemble_button': assemble_button,
            'individual_gallery': individual_gallery,
            'grid_gallery': grid_gallery,
            'mega_grid_image': mega_grid_image,
            # Include these as they are sometimes updated in the loop,
            # even though they are not explicitly outputs in the final UI update list
            'cfg_slider': cfg_slider,
            'sampler_dropdown': sampler_dropdown,
        }

        # Define the values passed from the UI
        gen_inputs = [
            positive_prompt, negative_prompt, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, width_slider, height_slider, seed_input,
            x_input, y_input, z_input
        ]
        
        # Define the explicit list of ALL components that run_xyz_matrix *might update*.
        # This list MUST exactly match the set of keys used in yielded dictionaries.
        # The order in this list does NOT matter when yielding dictionaries.
        gen_outputs_list = [
            html_log, generate_button, assemble_button, individual_gallery,
            grid_gallery, mega_grid_image, cfg_slider, sampler_dropdown
        ]
        
        # The functools.partial creates a new function where the first argument (components map) is pre-filled
        fn_with_components_for_gen = functools.partial(run_xyz_matrix, all_ui_components_map)
        
        # The 'outputs' parameter MUST be defined when the function yields dictionaries.
        # It must contain references to all the components that the function yields updates for.
        generate_button.click(fn=fn_with_components_for_gen, inputs=gen_inputs, outputs=gen_outputs_list)
        
        # Define the components that assemble_grids_from_last_run might update
        asm_outputs_list = [
            html_log, grid_gallery, mega_grid_image
        ]
        # For the assemble function, pass the same components map
        fn_with_components_for_asm = functools.partial(assemble_grids_from_last_run, all_ui_components_map)
        assemble_button.click(fn=fn_with_components_for_asm, inputs=None, outputs=asm_outputs_list)

    return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "XYZ Matrix POC", "xyz_matrix_poc_tab")]

on_ui_tabs(add_new_tab_to_ui)