#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.5.4
#
# Changelog:
# v0.5.4: Stability and Performance Update.
#         - Fixed significant pause between images by explicitly setting the VAE.
#         - Decoupled grid assembly into a separate button to prevent memory-related crashes.
# v0.5.3: Critical Bug Fix. Resolved UI loading deadlock.
#

import gradio as gr
import itertools
from PIL import Image, ImageDraw, ImageFont
from modules import shared, processing, sd_samplers, sd_schedulers
from modules.script_callbacks import on_ui_tabs

# --- Version Information ---
__version__ = "0.5.4"

# --- Globals to store data between button clicks ---
last_run_images = []
last_run_labels = {}

# --- Helper Functions ---
def create_labeled_grid(images, x_labels, y_labels, font_size=30, margin=5):
    # ... (This function remains unchanged)
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
    p_prompt, n_prompt, sampler, scheduler, steps, cfg, width, height, seed,
    x_values_str, y_values_str, z_values_str
):
    global last_run_images, last_run_labels
    last_run_images, last_run_labels = [], {}

    yield { html_log: "Parsing inputs...", generate_button: gr.Button.update(interactive=False), assemble_button: gr.Button.update(visible=False) }
    x_vals = [x.strip() for x in x_values_str.split('|') if x.strip()]
    y_vals = [y.strip() for y in y_values_str.split('|') if y.strip()]
    z_vals = [z.strip() for z in z_values_str.split('|') if z.strip()]
    total_images = len(x_vals) * len(y_vals) * len(z_vals)
    if total_images == 0:
        yield { html_log: "Error: No jobs to run.", generate_button: gr.Button.update(interactive=True) }; return
    
    job_list = list(itertools.product(z_vals, y_vals, x_vals))
    all_images = []
    
    yield { html_log: f"Starting generation of {total_images} images...", individual_gallery: None, grid_gallery: None, mega_grid_image: None }
    shared.state.begin(); shared.state.job_count = total_images
    for i, (z_val, y_val, x_val) in enumerate(job_list):
        if shared.state.interrupted: break
        shared.state.job = f"Image {i+1}/{total_images}"
        
        p = processing.StableDiffusionProcessingTxt2Img(
            sd_model=shared.sd_model,
            sd_vae=shared.sd_vae, # <-- FIX for the pause between images
            prompt=p_prompt.replace("{subject}", x_val), negative_prompt=n_prompt,
            steps=int(steps), cfg_scale=float(y_val), sampler_name=z_val, scheduler=scheduler,
            width=int(width), height=int(height), seed=int(seed) + i, n_iter=1, batch_size=1, do_not_save_samples=True,
        )
        proc = processing.process_images(p)
        if proc.images:
            all_images.append(proc.images[0])
            yield { individual_gallery: all_images, html_log: f"Generated image {i+1}/{total_images}" }
    shared.state.end()

    last_run_images = all_images
    last_run_labels = {'x': x_vals, 'y': y_vals, 'z': z_vals}
    
    yield {
        html_log: f"Generation of {len(all_images)} images complete. Ready to assemble grids.",
        generate_button: gr.Button.update(interactive=True),
        assemble_button: gr.Button.update(visible=True) # <-- Show the new button
    }

def assemble_grids_from_last_run():
    """This function is now separate, safer, and only does CPU work."""
    if not last_run_images or not last_run_labels:
        return { html_log: "No images from a previous run found to assemble." }

    x_vals, y_vals, z_vals = last_run_labels['x'], last_run_labels['y'], last_run_labels['z']
    
    # Grid Assembly
    yield { html_log: "Assembling X/Y grids..." }
    grid_images = []
    images_per_grid = len(x_vals) * len(y_vals)
    for i, z_label in enumerate(z_vals):
        grid_data = last_run_images[i * images_per_grid:(i + 1) * images_per_grid]
        grid = create_labeled_grid(grid_data, x_vals, y_vals)
        if grid: grid_images.append(grid)
    
    # Mega-Grid Assembly
    if len(grid_images) > 1:
        yield { grid_gallery: grid_images, html_log: "Assembling Mega-Grid..." }
        grid_w, grid_h = grid_images[0].size
        mega_grid = Image.new('RGB', (grid_w * len(grid_images), grid_h), 'white')
        for i, grid_img in enumerate(grid_images):
            mega_grid.paste(grid_img, (i * grid_w, 0))
        yield { mega_grid_image: mega_grid, html_log: "All grids assembled." }
    else:
        yield { mega_grid_image: grid_images[0] if grid_images else None, html_log: "Grid assembled." }

def create_streamer_ui():
    with gr.Blocks() as streamer_tab:
        # ... (UI Layout remains the same)
        gr.HTML(f"<h3>XYZ Matrix POC <span style='font-size:0.8rem;color:grey;'>v{__version__}</span></h3>")
        with gr.Accordion("Global Settings", open=True):
            # ... (all sliders and dropdowns are the same)
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
            # --- NEW: The Assemble Grids button ---
            assemble_button = gr.Button("Assemble Grids from Last Run", variant="secondary", visible=False)
            
        html_log = gr.HTML(label="Log")
        with gr.Tabs():
            with gr.TabItem("Live Individual Images"):
                individual_gallery = gr.Gallery(label="Individual Generations", show_label=False, columns=9)
            with gr.TabItem("X/Y Grids"):
                grid_gallery = gr.Gallery(label="X/Y Grids per Z-Value", show_label=False, columns=3)
            with gr.TabItem("Mega-Grid"):
                mega_grid_image = gr.Image(label="Final Mega-Grid", show_label=False)

        # --- Event Handlers ---
        gen_inputs = [
            positive_prompt, negative_prompt, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, width_slider, height_slider, seed_input,
            x_input, y_input, z_input
        ]
        gen_outputs = [individual_gallery, grid_gallery, mega_grid_image, html_log, generate_button, assemble_button]
        generate_button.click(fn=run_xyz_matrix, inputs=gen_inputs, outputs=gen_outputs)
        
        asm_outputs = [grid_gallery, mega_grid_image, html_log]
        assemble_button.click(fn=assemble_grids_from_last_run, inputs=None, outputs=asm_outputs)

    return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "XYZ Matrix POC", "xyz_matrix_poc_tab")]

on_ui_tabs(add_new_tab_to_ui)