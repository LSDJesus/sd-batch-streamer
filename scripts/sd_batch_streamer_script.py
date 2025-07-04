#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.6.5
#
# Changelog:
# v0.6.5: Extreme Memory Control for Grid Assembly.
#         - Images are now converted to in-memory bytes (PNG) after loading from disk
#           and before passing to grid creation, forcing aggressive PIL memory release.
#         - Grid creation re-loads images from these bytes buffers.
# v0.6.4: Critical Bug Fix. Resolved `SyntaxError`.
# v0.6.3: Critical Bug Fix. Resolved previous `SyntaxError`.
#

import gradio as gr
import itertools
import functools
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import tempfile
import gc # Import garbage collector
import io # For BytesIO

# --- Version Information ---
__version__ = "0.6.5"

# --- Globals to store data between button clicks ---
last_run_image_paths = [] # Paths to files saved on disk
last_run_labels = {}
temp_output_dir = None 

# --- Helper Functions ---
def create_labeled_grid(images_as_bytes, x_labels, y_labels, font_size=30, margin=5):
    # Load images from bytes for grid creation
    images = []
    for img_bytes in images_as_bytes:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            img.load() # Load pixel data
            images.append(img.copy()) # Create a copy to ensure our new object owns the data
            img.close() # Close BytesIO internal handle
            del img # Explicitly delete original PIL object
        except Exception as e:
            print(f"Error loading image from bytes for grid assembly: {e}")
            images.append(Image.new('RGB', (1,1), 'red')) # Placeholder
        gc.collect() # Aggressive GC

    if not all([images, x_labels, y_labels]):
        # Close any images that were loaded if there's an early exit
        for img_obj in images:
            if hasattr(img_obj, 'close'): img_obj.close()
            del img_obj
        return None
    
    grid_cols, grid_rows = len(x_labels), len(y_labels)
    if len(images) != grid_cols * grid_rows:
        print(f"Warning: Mismatch between image count ({len(images)}) and grid size ({grid_cols}x{grid_rows}) in create_labeled_grid.")
        # Close any images that were loaded if there's an early exit
        for img_obj in images:
            if hasattr(img_obj, 'close'): img_obj.close()
            del img_obj
        return None

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
        # Explicitly close and delete individual images after they are used for grid creation
        if hasattr(img, 'close'): img.close()
        del img
    del images # Delete the list of images
    gc.collect()

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
    global temp_output_dir
    
    html_log = ui_components_map['html_log']
    grid_gallery = ui_components_map['grid_gallery']
    mega_grid_image = ui_components_map['mega_grid_image']

    if not last_run_image_paths or not last_run_labels:
        yield { html_log: gr.HTML.update(value="No images from a previous run found to assemble.") }; return

    x_vals, y_vals, z_vals = last_run_labels['x'], last_run_labels['y'], last_run_labels['z']
    
    yield { html_log: gr.HTML.update(value="Assembling X/Y grids...") }
    grid_images_for_display = [] # For storing the final grid PIL objects for display
    images_per_grid = len(x_vals) * len(y_vals)
    
    # Load images from disk, convert to bytes, and discard PIL objects immediately
    images_as_bytes_list = []
    for img_path in last_run_image_paths:
        try:
            with Image.open(img_path) as img:
                with io.BytesIO() as buffer:
                    img.save(buffer, format="PNG") # Save as PNG bytes
                    images_as_bytes_list.append(buffer.getvalue())
            del img # Ensure PIL object is deleted
        except Exception as e:
            print(f"Error loading image {img_path} to bytes: {e}")
            images_as_bytes_list.append(None) # Store None for failed loads
        gc.collect() # Aggressive GC after each conversion

    # Create grids from bytes data
    for i, z_label in enumerate(z_vals):
        grid_data_bytes = images_as_bytes_list[i * images_per_grid:(i + 1) * images_per_grid]
        # Filter out None values if any images failed to load
        grid_data_bytes = [b for b in grid_data_bytes if b is not None] 

        grid = create_labeled_grid(grid_data_bytes, x_vals, y_vals)
        
        if grid: 
            grid_images_for_display.append(grid)
            yield { grid_gallery: gr.Gallery.update(value=grid_images_for_display), html_log: gr.HTML.update(value=f"Assembled grid for Z={z_label}...") }
            gc.collect() # Trigger garbage collection after yielding a grid

    # After all individual images are loaded and used, clear the list of paths to free references
    last_run_image_paths.clear()
    images_as_bytes_list.clear() # Also clear the bytes list
    gc.collect()

    if len(grid_images_for_display) > 1:
        yield { html_log: gr.HTML.update(value="Assembling Mega-Grid...") }
        grid_w, grid_h = grid_images_for_display[0].size
        mega_grid = Image.new('RGB', (grid_w * len(grid_images_for_display), grid_h), 'white')
        for i, grid_img in enumerate(grid_images_for_display):
            mega_grid.paste(grid_img, (i * grid_w, 0))
            if hasattr(grid_img, 'close'): grid_img.close() # Close sub-grids
            del grid_img # Delete sub-grid object
        del grid_images_for_display # Delete the list of sub-grids
        gc.collect()
        yield { mega_grid_image: gr.Image.update(value=mega_grid), html_log: gr.HTML.update(value="All grids assembled.") }
    else:
        yield { grid_gallery: gr.Gallery.update(value=None), mega_grid_image: gr.Image.update(value=grid_images_for_display[0] if grid_images_for_display else None), html_log: gr.HTML.update(value="Grid assembled.") }
        if len(grid_images_for_display) == 1 and hasattr(grid_images_for_display[0], 'close'):
             grid_images_for_display[0].close() # Close the single grid
        del grid_images_for_display
        gc.collect()

    # Attempt to clean up the temporary directory after everything is done
    if temp_output_dir and os.path.exists(temp_output_dir):
        try:
            shutil.rmtree(temp_output_dir)
            temp_output_dir = None
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
            'cfg_slider': cfg_slider, 'sampler_dropdown': sampler_dropdown,
        }

        gen_inputs = [
            positive_prompt, negative_prompt, sampler_dropdown, scheduler_dropdown, steps_slider, cfg_slider, width_slider, height_slider, seed_input,
            x_input, y_input, z_input
        ]
        
        gen_outputs_list = [
            html_log, generate_button, assemble_button, individual_gallery, grid_gallery, mega_grid_image
        ]
        
        fn_with_components_for_gen = functools.partial(run_xyz_matrix, all_ui_components_map)
        generate_button.click(fn=fn_with_components_for_gen, inputs=gen_inputs, outputs=gen_outputs_list)
        
        asm_outputs_list = [
            html_log, grid_gallery, mega_grid_image
        ]
        fn_with_components_for_asm = functools.partial(assemble_grids_from_last_run, all_ui_components_map)
        assemble_button.click(fn=fn_with_components_for_asm, inputs=None, outputs=asm_outputs_list)

    return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "XYZ Matrix POC", "xyz_matrix_poc_tab")]

on_ui_tabs(add_new_tab_to_ui)