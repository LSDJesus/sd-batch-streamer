#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.6.2
#
# Changelog:
# v0.6.2: Critical Stability Fix for Grid Assembly.
#         - Implemented aggressive PIL image memory management during grid assembly:
#           Explicitly loads pixel data, deep copies for use, and deletes references.
#         - Added intermediate yields during grid assembly to provide better feedback and aid memory management.
# v0.6.1: Memory Optimization for Grid Assembly (initial attempt).
# v0.6.0: Final Stability Fix for UI loading/event handlers.
#

import gradio as gr
import itertools
import functools
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import tempfile
import gc # Import garbage collector

# --- Version Information ---
__version__ = "0.6.2"

# --- Globals to store data between button clicks ---
last_run_image_paths = []
last_run_labels = {}
temp_output_dir = None 

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
    
    global last_run_image_paths, last_run_labels, temp_output_dir
    last_run_image_paths, last_run_labels = [], {}

    # Clean up previous temporary directory if it exists
    if temp_output_dir and os.path.exists(temp_output_dir):
        try:
            shutil.rmtree(temp_output_dir)
        except OSError as e:
            print(f"Error removing old temporary directory {temp_output_dir}: {e}")
    
    # Create a new temporary directory for this run's images
    temp_output_dir = tempfile.mkdtemp(prefix="sd-matrix-")

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
    live_gallery_images = [] # This list holds PIL objects only for the live preview
    
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
            new_image = proc.images[0]
            live_gallery_images.append(new_image) # Add to list for live preview

            # --- Save image to disk and store path ---
            img_filename = os.path.join(temp_output_dir, f"img_{i:03d}.png")
            new_image.save(img_filename)
            last_run_image_paths.append(img_filename)
            
            yield { individual_gallery: gr.Gallery.update(value=live_gallery_images), html_log: gr.HTML.update(value=f"Generated image {i+1}/{total_images}") }
            
            # Explicitly clear references to the generated PIL image to help garbage collection
            new_image = None
            proc = None
            gc.collect() # Trigger garbage collection

    shared.state.end()

    last_run_labels = {'x': x_vals, 'y': y_vals, 'z': z_vals}
    
    yield {
        html_log: gr.HTML.update(value=f"Generation of {len(last_run_image_paths)} images complete. Ready to assemble grids."),
        generate_button: gr.Button.update(interactive=True),
        assemble_button: gr.Button.update(visible=True)
    }

def assemble_grids_from_last_run(ui_components_map):
    html_log = ui_components_map['html_log']
    grid_gallery = ui_components_map['grid_gallery']
    mega_grid_image = ui_components_map['mega_grid_image']

    if not last_run_image_paths or not last_run_labels:
        yield { html_log: gr.HTML.update(value="No images from a previous run found to assemble.") }; return

    x_vals, y_vals, z_vals = last_run_labels['x'], last_run_labels['y'], last_run_labels['z']
    
    yield { html_log: gr.HTML.update(value="Assembling X/Y grids...") }
    grid_images = []
    images_per_grid = len(x_vals) * len(y_vals)
    
    # Load images from disk as needed for grid assembly, and explicitly manage their memory
    # We load them in chunks for each grid, not all at once.
    for i, z_label in enumerate(z_vals):
        grid_data_paths = last_run_image_paths[i * images_per_grid:(i + 1) * images_per_grid]
        current_grid_images = []
        for img_path in grid_data_paths:
            try:
                img = Image.open(img_path)
                img.load() # Load image data into memory
                current_grid_images.append(img.copy()) # Create a copy to ensure data is in our new object
                img.close() # Close original file handle
                del img # Explicitly delete original PIL object
            except Exception as e:
                print(f"Error loading image {img_path} for grid assembly: {e}")
                current_grid_images.append(Image.new('RGB', (1,1), 'red')) # Small red placeholder
            gc.collect() # Trigger garbage collection after each image load

        grid = create_labeled_grid(current_grid_images, x_vals, y_vals)
        
        # Explicitly close and delete individual images after they are used for grid creation
        for img_obj in current_grid_images:
            if hasattr(img_obj, 'close'): img_obj.close()
            del img_obj
        del current_grid_images
        gc.collect()

        if grid: 
            grid_images.append(grid)
            yield { grid_gallery: gr.Gallery.update(value=grid_images), html_log: gr.HTML.update(value=f"Assembled grid for Z={z_label}...") }
            gc.collect() # Trigger garbage collection after yielding a grid

    # After all individual images are loaded and used, clear the list of paths to free references
    last_run_image_paths.clear()

    if len(grid_images) > 1:
        yield { html_log: gr.HTML.update(value="Assembling Mega-Grid...") }
        grid_w, grid_h = grid_images[0].size
        mega_grid = Image.new('RGB', (grid_w * len(grid_images), grid_h), 'white')
        for i, grid_img in enumerate(grid_images):
            mega_grid.paste(grid_img, (i * grid_w, 0))
            if hasattr(grid_img, 'close'): # Close the sub-grids after they are pasted
                grid_img.close()
            del grid_img # Delete the sub-grid object
        del grid_images # Delete the list of sub-grids
        gc.collect()
        yield { mega_grid_image: gr.Image.update(value=mega_grid), html_log: gr.HTML.update(value="All grids assembled.") }
    else:
        # Clear grid_gallery if only one grid (no actual gallery display)
        yield { grid_gallery: gr.Gallery.update(value=None), mega_grid_image: gr.Image.update(value=grid_images[0] if grid_images else None), html_log: gr.HTML.update(value="Grid assembled.") }
        if len(grid_images) == 1 and hasattr(grid_images[0], 'close'):
             grid_images[0].close() # Close the single grid if only one
        del grid_images
        gc.collect()

    # Attempt to clean up the temporary directory after everything is done
    if temp_output_dir and os.path.exists(temp_output_dir):
        try:
            shutil.rmtree(temp_output_dir)
            global temp_output_dir
            temp_output_dir = None # Clear the global reference
        except OSError as e:
            print(f"Error removing temporary directory {temp_output_dir} after assembly: {e}")
    
    gc.collect() # Final garbage collection

# --- UI Layout (Remains unchanged) ---
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

        # --- Event Handlers ---
        all_ui_components_map = {
            'html_log': html_log, 'generate_button': generate_button, 'assemble_button': assemble_button,
            'individual_gallery': individual_gallery, 'grid_gallery': grid_gallery, 'mega_grid_image': mega_grid_image,
            'cfg_slider': cfg_slider, 'sampler_dropdown': sampler_dropdown, # UI objects needed for dynamic updating
        }

        gen_inputs = [
            positive_prompt, negative_prompt, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, width_slider, height_slider, seed_input,
            x_input, y_input, z_input
        ]
        
        gen_outputs_list = [
            html_log, generate_button, assemble_button, individual_gallery, grid_gallery, mega_grid_image # Must include all potentially updated components
        ]
        
        fn_with_components_for_gen = functools.partial(run_xyz_matrix, all_ui_components_map)
        generate_button.click(fn=fn_with_components_for_gen, inputs=gen_inputs, outputs=gen_outputs_list)
        
        asm_outputs_list = [
            html_log, grid_gallery, mega_grid_image # Must include all potentially updated components
        ]
        fn_with_components_for_asm = functools.partial(assemble_grids_from_last_run, all_ui_components_map)
        assemble_button.click(fn=fn_with_components_for_asm, inputs=None, outputs=asm_outputs_list)

    return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "XYZ Matrix POC", "xyz_matrix_poc_tab")]

on_ui_tabs(add_new_tab_to_ui)