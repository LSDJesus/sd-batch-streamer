import gradio as gr
import modules.scripts as scripts
from modules import shared, processing
import time

# We inherit from the scripts.Script class.
class BatchStreamerScript(scripts.Script):

    # The title of the script in the UI.
    def title(self):
        return "Batch Image Streamer"

    # This script should only be shown in the txt2img tab.
    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    # This is where we define the UI for our script.
    def ui(self, is_img2img):
        # We start with a collapsible group.
        with gr.Accordion(self.title(), open=False):
            with gr.Row():
                # This checkbox is the standard way to enable/disable a script.
                # However, for our custom button approach, it's more of a toggle
                # to show/hide the UI components. We'll use our own button.
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
                    label="Live Output", show_label=True, elem_id="batch_stream_gallery"
                ).style(columns=[4], rows=[2], object_fit="contain", height="auto")

        # This is the core of the solution. We define a Python generator function
        # that will handle the image generation and 'yield' results.
        def process_and_stream_images(prompts_text, steps_val, cfg_val):
            # 1. Clear the gallery at the beginning of the process.
            # We yield an empty list to the gallery component.
            yield {output_gallery: []}
            
            # 2. Parse the input prompts. Ignore empty lines.
            prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
            if not prompts:
                print("Batch Streamer: No prompts provided.")
                # We must yield a final state for the output component.
                yield {output_gallery: []}
                return

            print(f"Batch Streamer: Starting generation for {len(prompts)} prompts.")
            
            # This list will hold all generated images.
            all_images = []

            # We need to temporarily disable the main UI's generate button to prevent
            # confusion and double-clicks. We will re-enable it at the end.
            # This is an advanced Gradio feature.
            yield {stream_button: gr.Button.update(value="Generating...", interactive=False)}

            # 3. Loop through each prompt.
            for i, prompt in enumerate(prompts):
                print(f"Batch Streamer: Processing prompt {i+1}/{len(prompts)}: '{prompt}'")
                
                # IMPORTANT: We create a new processing object for each image.
                # This ensures settings are fresh for each run. We grab the
                # currently loaded model from the shared state.
                p = processing.StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model,
                    prompt=prompt,
                    # We can grab other settings from the main UI if we want,
                    # or use the ones from our custom UI.
                    negative_prompt=shared.opts.data.get("negative_prompt", ""),
                    steps=int(steps_val),
                    cfg_scale=float(cfg_val),
                    sampler_name='Euler a', # Or make this a dropdown in your UI
                    seed=-1,
                    width=shared.opts.data.get("width", 512),
                    height=shared.opts.data.get("height", 512),
                    n_iter=1,
                    batch_size=1,
                )

                # Run the actual image generation process.
                processed = processing.process_images(p)
                
                # The result is in processed.images. Add the first one to our list.
                if processed.images:
                    new_image = processed.images[0]
                    all_images.append(new_image)
                
                    # 4. The MAGIC! Yield the updated list of images.
                    # Gradio receives this and automatically updates the 'output_gallery'.
                    # We send a dictionary mapping the component to its new value.
                    yield {output_gallery: all_images}
            
            print("Batch Streamer: All prompts processed.")
            # Re-enable our button now that the process is complete.
            yield {
                output_gallery: all_images,
                stream_button: gr.Button.update(value="Generate and Stream", interactive=True)
            }

        # 5. Connect our button to our function.
        # When 'stream_button' is clicked, it calls 'process_and_stream_images'.
        # The 'inputs' are the Gradio components whose values are passed to the function.
        # The 'outputs' are the components that the function will update via 'yield'.
        stream_button.click(
            fn=process_and_stream_images,
            inputs=[prompts_input, steps, cfg_scale],
            outputs=[output_gallery, stream_button]
        )

        # The 'ui' method must return a list of components that will be passed to the 'run' method.
        # Since we are not using the main generate button or the 'run' method,
        # we can just return the enable checkbox.
        return [enabled]

    # We are not using the standard 'run' method because it doesn't support yielding.
    # We've built our own logic tied to our custom button. So, this method does nothing.
    def run(self, p, enabled):
        # If you were to press the MAIN Generate button while this script is active,
        # this method would be called. We'll make it do nothing.
        if not enabled:
            return
        
        # We return an empty Processed object to avoid any errors.
        return processing.Processed(p, [], p.seed, "")