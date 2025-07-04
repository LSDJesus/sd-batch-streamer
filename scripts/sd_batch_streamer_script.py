#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.5.2
#
# Changelog:
# v0.5.2: Bug fix. Removed conflicting 'outputs' argument from the main click handler, which was causing an infinite UI load.
# v0.5.1: Bug fix. Correctly import 'sd_schedulers' module.
# v0.5.0: Proof of Concept. Added X/Y/Z Matrix functionality.
#

import gradio as gr
import itertools
from PIL import Image, ImageDraw, ImageFont
from modules import shared, processing, sd_samplers, sd_schedulers
from modules.script_callbacks import on_ui_tabs

# --- Version Information ---
__version__ = "0.5.2"

# --- Helper Functions ---

def create_labeled_grid(images, x_labels, y_labels, font_size=30, margin=5):
    """Creates a grid of images with labels for X and Y axes."""
    if not images or not x_labels or not y_labels: return None
    grid_cols, grid_rows = len(x_labels), len(y_labels)
    if len(images) != grid_cols * grid_rows: return None
    img_w, img_h = images[0].size
    label_area_size = font_size + 2 * margin
    grid_w, grid_h = img_w * grid_cols + label_area_size, img_h * grid_rows + label_area_size
    grid_image = Image.new('RGB', (grid_w, grid_h), 'white')
    draw = ImageDraw.Draw(grid_image)
    try: font = ImageFont.truetype("arial.ttf", font_size)
    except IOError: font = ImageFont.load_default()
    for i, label in enumerate(y_labels):
        draw.text((margin, label_area_size + i * img_h + img_h // 2 - font_size // 2), str(label), fill="black", font=font)
    for i, label in enumerate(x_labels):
        draw.text((label_area_size + i * img_w + img_w // 2 - len(str(label)) * font_size // 4, margin), str(label), fill="black", font=font)
    for i, img in enumerate(images):
        grid_image.paste(img, (label_area_size + (i % grid_cols) * img_w, label_area_size + (i // grid_cols) * img_h))
    return grid_image

def create_streamer_ui():
    """Creates the Gradio UI components for our proof-of-concept tab."""
    with gr.Blocks() as streamer_tab:
        
        def run_xyz_matrix(
            p_prompt, n_prompt, sampler, scheduler, steps, cfg, width, height, seed,
            x_values_str, y_values_str, z_values_str
        ):
            yield { html_log: "Phase 1: Parsing inputs and creating job list...", generate_button: gr.Button.update(interactive=False) }
            x_values = [x.strip() for x in x_values_str.split('|')]
            y_values = [y.strip() for y in y_values_str.split('|')]
            z_values = [z.strip() for z in z_values_str.split('|')]
            total_images = len(x_values) * len(y_values) * len(z_values)
            if total_images == 0:
                yield { html_log: "Error: No jobs to run. Check inputs.", generate_button: gr.Button.update(interactive=True) }; return
            job_list = list(itertools.product(z_values, y_values, x_values))
            all_individual_images = []
            
            yield { html_log: f"Phase 2: Starting generation of {total_images} images..." }
            shared.state.begin(); shared.state.job_count = total_images
            for i, (z_val, y_val, x_val) in enumerate(job_list):
                if shared.state.interrupted: break
                shared.state.job = f"Image {i+1}/{total_images}"
                p_copy = processing.StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model, prompt=p_prompt.replace("{subject}", x_val), negative_prompt=n_prompt,
                    steps=int(steps), cfg_scale=float(y_val), sampler_name=z_val, scheduler=scheduler,
                    width=int(width), height=int(height), seed=int(seed) + i, n_iter=1, batch_size=1, do_not_save_samples=True,
                )
                processed_single = processing.process_images(p_copy)
                if processed_single.images:
                    new_image = processed_single.images[0]
                    all_individual_images.append(new_image)
                    yield {
                        individual_gallery: all_individual_images,
                        html_log: f"Phase 2: Generated image {i+1}/{total_images} (Subject: {x_val}, CFG: {y_val}, Sampler: {z_val})",
                        cfg_slider: gr.Slider.update(value=float(y_val)),
                        sampler_dropdown: gr.Dropdown.update(value=z_val)
                    }
            shared.state.end()
            
            yield { html_log: "Phase 3: Assembling X/Y grids..." }
            final_grid_images = []
            num_images_per_grid = len(x_values) * len(y_values)
            for i, z_label in enumerate(z_values):
                images_for_grid = all_individual_images[i * num_images_per_grid:(i + 1) * num_images_per_grid]
                grid = create_labeled_grid(images_for_grid, x_values, y_values)
                if grid: final_grid_images.append(grid)
            yield { grid_gallery: final_grid_images, html_log: "Phase 3: X/Y grids assembled." }
            
            if len(final_grid_images) > 1:
                yield { html_log: "Phase 4: Assembling Mega-Grid..." }
                grid_w, grid_h = final_grid_images[0].size
                mega_grid = Image.new('RGB', (grid_w * len(final_grid_images), grid_h), 'white')
                for i, grid_img in enumerate(final_grid_images):
                    mega_grid.paste(grid_img, (i * grid_w, 0))
                yield { mega_grid_image: mega_grid, html_log: "All phases complete." }
            else:
                 yield { mega_grid_image: final_grid_images[0] if final_grid_images else None, html_log: "All phases complete." }
            yield { generate_button: gr.Button.update(interactive=True) }
            
        # --- UI Layout ---
        with gr.Blocks() as streamer_tab:
            gr.HTML(f"<h3>XYZ Matrix Proof of Concept <span style='font-size:0.8rem;color:grey;'>v{__version__}</span></h3>")
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
            generate_button = gr.Button("Generate XYZ Matrix", variant="primary")
            html_log = gr.HTML(label="Log")
            with gr.Tabs():
                with gr.TabItem("Live Individual Images"):
                    individual_gallery = gr.Gallery(label="Individual Generations", show_label=False, columns=9)
                with gr.TabItem("X/Y Grids"):
                    grid_gallery = gr.Gallery(label="X/Y Grids per Z-Value", show_label=False, columns=3)
                with gr.TabItem("Mega-Grid"):
                    mega_grid_image = gr.Image(label="Final Mega-Grid", show_label=False)

        # --- Event Handlers ---
        inputs = [
            positive_prompt, negative_prompt, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, width_slider, height_slider, seed_input,
            x_input, y_input, z_input
        ]
        
        # --- FIX: Removed the 'outputs' argument from this line ---
        generate_button.click(fn=run_xyz_matrix, inputs=inputs, outputs=None)

    return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "XYZ Matrix POC", "xyz_matrix_poc_tab")]

on_ui_tabs(add_new_tab_to_ui)