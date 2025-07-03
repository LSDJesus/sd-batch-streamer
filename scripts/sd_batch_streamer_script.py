import gradio as gr
import modules.scripts as scripts
from modules import shared, processing
import time

# --- CHANGE 1: Updated the class name for clarity and consistency. ---
class SDBatchStreamerScript(scripts.Script):

    # --- CHANGE 2: Updated the title that appears in the UI. ---
    def title(self):
        return "SD Batch Streamer"

    # This script should only be shown in the txt2img tab.
    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    # This is where we define the UI for our script.
    def ui(self, is_img2img):
        # We start with a collapsible group.
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                # This checkbox is the standard way to enable/disable a script.
                enabled = gr.Checkbox(label="Enable (required to see UI, but use button below)", value=True)
            
            with gr.Row():
                prompts_input = gr.Textbox(
                    label="Prompts (one per line)",
                    lines=5,
                    placeholder="A beautiful cat\nA majestic dog\nA futuristic car"
                )

            with gr.Row():
                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Steps", value=20)
                cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)
            
            with gr.Row():
                # This is OUR button. Clicking it will trigger our custom function.
                stream_button = gr.Button("Generate and Stream", variant="primary")
            
            with gr.Row():
                # This is the gallery where images will be displayed in real-time.
                output_gallery = gr.Gallery(
                    label="Live Output", show_label=True, elem_id="sd_batch_stream_gallery"
                ).style(columns=[4], rows=[2], object_fit="contain", height="auto")

        # The core of the solution: a generator function.
        def process_and_stream_images(prompts_text, steps_val, cfg_val):
            yield {output_gallery: []}
            
            prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
            if not prompts:
                print("SD Batch Streamer: No prompts provided.")
                yield {output_gallery: []}
                return

            print(f"SD Batch Streamer: Starting generation for {len(prompts)} prompts.")
            
            all_images = []

            yield {stream_button: gr.Button.update(value="Generating...", interactive=False)}

            for i, prompt in enumerate(prompts):
                print(f"SD Batch Streamer: Processing prompt {i+1}/{len(prompts)}: '{prompt}'")
                
                p = processing.StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model,
                    prompt=prompt,
                    negative_prompt=shared.opts.data.get("negative_prompt", ""),
                    steps=int(steps_val),
                    cfg_scale=float(cfg_val),
                    sampler_name='Euler a',
                    seed=-1,
                    width=shared.opts.data.get("width", 512),
                    height=shared.opts.data.get("height", 512),
                    n_iter=1,
                    batch_size=1,
                )

                processed = processing.process_images(p)
                
                if processed.images:
                    new_image = processed.images[0]
                    all_images.append(new_image)
                    yield {output_gallery: all_images}
            
            print("SD Batch Streamer: All prompts processed.")
            yield {
                output_gallery: all_images,
                stream_button: gr.Button.update(value="Generate and Stream", interactive=True)
            }

        stream_button.click(
            fn=process_and_stream_images,
            inputs=[prompts_input, steps, cfg_scale],
            outputs=[output_gallery, stream_button]
        )

        return [enabled]

    def run(self, p, enabled):
        if not enabled:
            return
        return processing.Processed(p, [], p.seed, "")