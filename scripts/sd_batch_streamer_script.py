#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.3.0
#
# Changelog:
# v0.3.0: Feature. Added "Send to inputs" functionality. Click an image in the gallery to load its settings.
# v0.2.3: Bug fix. Replaced deprecated .style() method for gr.Gallery.
# v0.2.2: Bug fix. Corrected import to use 'modules.sd_samplers'.
# v0.2.1: Bug fix. Attempted to correct sampler loading.
# v0.2.0: Major revamp. Converted the extension into a standalone top-level UI tab.
# v0.1.0: Initial release as a script within the txt2img tab.
#

import gradio as gr
import json
from modules import shared, processing, sd_samplers
from modules.script_callbacks import on_ui_tabs

# --- Version Information ---
__version__ = "0.3.0"

# --- Globals for this script ---
# We need a place to store the parameters for each generated image.
# A simple dictionary mapping index to parameters will work.
image_params_storage = {}

def create_streamer_ui():
    """
    Creates the Gradio UI components for our tab.
    """
    with gr.Blocks() as streamer_tab:
        
        # The main processing logic function
        def process_and_stream_images(prompts_text, negative_prompt, steps_val, cfg_val, width, height, sampler_name):
            # Clear storage at the start of a new batch
            global image_params_storage
            image_params_storage.clear()

            yield {output_gallery: gr.Gallery.update(value=[], visible=True)}
            
            prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
            if not prompts:
                print("SD Batch Streamer: No prompts provided.")
                yield {output_gallery: []}
                return

            print(f"SD Batch Streamer (v{__version__}): Starting generation for {len(prompts)} prompts.")
            all_images = []
            
            yield {stream_button: gr.Button.update(value="Generating...", interactive=False)}
            
            shared.state.begin()
            shared.state.job_count = len(prompts)

            for i, prompt in enumerate(prompts):
                shared.state.job = f"Prompt: {prompt[:80]}..."
                shared.state.job_no = i + 1
                if shared.state.interrupted:
                    break

                print(f"Processing prompt {i+1}/{len(prompts)}: '{prompt}'")
                
                p = processing.StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=int(steps_val),
                    cfg_scale=float(cfg_val),
                    sampler_name=sampler_name,
                    seed=-1, # We will get the real seed after processing
                    width=int(width),
                    height=int(height),
                    n_iter=1,
                    batch_size=1,
                )

                processed = processing.process_images(p)
                
                if processed.images:
                    new_image = processed.images[0]
                    all_images.append(new_image)
                    
                    # --- NEW: Store parameters for the generated image ---
                    # We use the current image index 'i' as the key.
                    # We retrieve the actual seed used from the processed object.
                    image_params_storage[i] = {
                        "prompt": prompt,
                        "negative_prompt": negative_prompt,
                        "steps": steps_val,
                        "cfg_scale": cfg_val,
                        "width": width,
                        "height": height,
                        "sampler_name": sampler_name,
                        "seed": processed.seed
                    }
                    
                    yield {output_gallery: all_images}
            
            shared.state.end()
            print("SD Batch Streamer: All prompts processed.")
            yield {
                output_gallery: all_images,
                stream_button: gr.Button.update(value="Generate and Stream", interactive=True)
            }

        # --- NEW: Function to handle gallery clicks ---
        def on_gallery_select(evt: gr.SelectData, prompts_text, neg_prompt_text):
            # evt.index is the index of the clicked image in the gallery
            params = image_params_storage.get(evt.index)
            
            if params:
                # To avoid replacing the whole prompt list, we can prepend the selected prompt.
                # A more user-friendly approach might be to just replace the first line.
                updated_prompts = f"{params['prompt']}\n"
                
                # We return a dictionary mapping components to their new values.
                return {
                    prompts_input: updated_prompts,
                    negative_prompt: params["negative_prompt"],
                    steps: params["steps"],
                    cfg_scale: params["cfg_scale"],
                    width: params["width"],
                    height: params["height"],
                    sampler: params["sampler_name"]
                }
            
            # If no params found, return original values
            return {
                prompts_input: prompts_text,
                negative_prompt: neg_prompt_text,
            }

        # --- UI Layout ---
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(f"<h3>SD Batch Streamer <span style='font-size:0.8rem;color:grey;'>v{__version__}</span></h3>")
                gr.HTML("<p>Click an image in the gallery to send its settings back to the controls.</p>")
                
                prompts_input = gr.Textbox(label="Prompts (one per line)", lines=8, placeholder="A beautiful cat\nA majestic dog...")
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=3, placeholder="ugly, deformed...")
                
                with gr.Row():
                    stream_button = gr.Button("Generate and Stream", variant="primary")
                    interrupt_button = gr.Button("Interrupt", variant="secondary")
                    skip_button = gr.Button("Skip", variant="secondary")

                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=150, step=1, label="Steps", value=20)
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)

                with gr.Row():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                
                sampler_choices = [s.name for s in sd_samplers.all_samplers]
                sampler = gr.Dropdown(label='Sampling method', choices=sampler_choices, value='Euler a')

            with gr.Column(scale=3):
                gr.HTML("<h4>Live Output</h4>")
                output_gallery = gr.Gallery(
                    label="Live Output", show_label=False, elem_id="sd_batch_stream_gallery",
                    columns=4, rows=2, object_fit="contain", height="auto"
                )

        # --- Event Handlers ---
        stream_button.click(
            fn=process_and_stream_images,
            inputs=[prompts_input, negative_prompt, steps, cfg_scale, width, height, sampler],
            outputs=[output_gallery, stream_button]
        )

        interrupt_button.click(fn=lambda: shared.state.interrupt(), inputs=None, outputs=None)
        skip_button.click(fn=lambda: shared.state.skip(), inputs=None, outputs=None)
        
        # --- NEW: Connect the gallery's select event to our handler function ---
        gallery_outputs = [prompts_input, negative_prompt, steps, cfg_scale, width, height, sampler]
        output_gallery.select(
            fn=on_gallery_select,
            inputs=[prompts_input, negative_prompt],
            outputs=gallery_outputs
        )

        return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "Batch Streamer", "sd_batch_streamer_tab")]

on_ui_tabs(add_new_tab_to_ui)