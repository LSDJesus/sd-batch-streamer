#
# SD Batch Streamer
#
# Author: LSDJesus
# Version: v0.2.1
#
# Changelog:
# v0.2.1: Bug fix. Correctly load samplers from modules.samplers instead of modules.shared.
# v0.2.0: Major revamp. Converted the extension into a standalone top-level UI tab.
# v0.1.0: Initial release as a script within the txt2img tab.
#

import gradio as gr
from modules import shared, processing, samplers # --- FIX: Import 'samplers' module ---
from modules.script_callbacks import on_ui_tabs
from modules.ui_components import ToolButton

# --- Version Information ---
__version__ = "0.2.1"

def create_streamer_ui():
    """
    Creates the Gradio UI components for our tab.
    This function is called once to build the layout.
    """
    with gr.Blocks() as streamer_tab:
        # The main processing logic function
        def process_and_stream_images(prompts_text, negative_prompt, steps_val, cfg_val, width, height, sampler_name):
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
                    seed=-1,
                    width=int(width),
                    height=int(height),
                    n_iter=1,
                    batch_size=1,
                )

                processed = processing.process_images(p)
                
                if processed.images:
                    new_image = processed.images[0]
                    all_images.append(new_image)
                    yield {output_gallery: all_images}
            
            shared.state.end()
            print("SD Batch Streamer: All prompts processed.")
            yield {
                output_gallery: all_images,
                stream_button: gr.Button.update(value="Generate and Stream", interactive=True)
            }

        # --- UI Layout ---
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(f"<h3>SD Batch Streamer <span style='font-size:0.8rem;color:grey;'>v{__version__}</span></h3>")
                
                prompts_input = gr.Textbox(
                    label="Prompts (one per line)",
                    lines=8,
                    placeholder="A beautiful cat\nA majestic dog\nA futuristic car"
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    lines=3,
                    placeholder="ugly, deformed, bad anatomy..."
                )
                
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
                
                # --- FIX: Get sampler choices from the correct module ---
                sampler_choices = [s.name for s in samplers.all_samplers]
                sampler = gr.Dropdown(label='Sampling method', choices=sampler_choices, value='Euler a')

            with gr.Column(scale=3):
                gr.HTML("<h4>Live Output</h4>")
                output_gallery = gr.Gallery(
                    label="Live Output", show_label=False, elem_id="sd_batch_stream_gallery"
                ).style(columns=[4], rows=[2], object_fit="contain", height="auto")

        # --- Event Handlers ---
        stream_button.click(
            fn=process_and_stream_images,
            inputs=[prompts_input, negative_prompt, steps, cfg_scale, width, height, sampler],
            outputs=[output_gallery, stream_button]
        )

        interrupt_button.click(fn=lambda: shared.state.interrupt(), inputs=None, outputs=None)
        skip_button.click(fn=lambda: shared.state.skip(), inputs=None, outputs=None)

        return streamer_tab

def add_new_tab_to_ui():
    new_tab = create_streamer_ui()
    return [(new_tab, "Batch Streamer", "sd_batch_streamer_tab")]

on_ui_tabs(add_new_tab_to_ui)