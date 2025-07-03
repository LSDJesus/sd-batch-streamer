import gradio as gr
from modules import shared, processing
from modules.script_callbacks import on_ui_tabs
from modules.ui_components import ToolButton

# We no longer use the 'scripts.Script' class.
# This code will be executed when the UI is being built.

def create_streamer_ui():
    """
    Creates the Gradio UI components for our tab.
    This function is called once to build the layout.
    """
    # We start with a Gradio 'Blocks' context, which is the top-level container.
    with gr.Blocks() as streamer_tab:
        # The main processing logic function, now with more parameters for a self-contained UI.
        def process_and_stream_images(prompts_text, negative_prompt, steps_val, cfg_val, width, height, sampler):
            # 1. Clear the gallery and yield the update.
            yield {output_gallery: gr.Gallery.update(value=[], visible=True)}
            
            prompts = [p.strip() for p in prompts_text.splitlines() if p.strip()]
            if not prompts:
                print("SD Batch Streamer: No prompts provided.")
                yield {output_gallery: []}
                return

            print(f"SD Batch Streamer: Starting generation for {len(prompts)} prompts.")
            all_images = []
            
            # Disable the button during processing
            yield {stream_button: gr.Button.update(value="Generating...", interactive=False)}
            
            # Set a shared state variable to indicate we are running.
            shared.state.begin()

            # 3. Loop through each prompt.
            for i, prompt in enumerate(prompts):
                # Check for interruption requests from the main UI (e.g., Skip/Interrupt buttons)
                shared.state.job = f"Batch Streamer: {i+1}/{len(prompts)}"
                shared.state.job_no = i + 1
                if shared.state.interrupted:
                    break

                print(f"SD Batch Streamer: Processing prompt {i+1}/{len(prompts)}: '{prompt}'")
                
                # Create the processing object with values from OUR UI.
                p = processing.StableDiffusionProcessingTxt2Img(
                    sd_model=shared.sd_model,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    steps=int(steps_val),
                    cfg_scale=float(cfg_val),
                    sampler_name=sampler,
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
                    # The MAGIC: Yield the updated list of images to the gallery.
                    yield {output_gallery: all_images}
            
            # Reset the state and re-enable the button.
            shared.state.end()
            print("SD Batch Streamer: All prompts processed.")
            yield {
                output_gallery: all_images,
                stream_button: gr.Button.update(value="Generate and Stream", interactive=True)
            }

        # --- UI Layout ---
        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("<h3>SD Batch Streamer</h3>")
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
                    # We can add the standard Interrupt and Skip buttons for better integration.
                    interrupt_button = gr.Button("Interrupt", variant="secondary")
                    skip_button = gr.Button("Skip", variant="secondary")

                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=150, step=1, label="Steps", value=20)
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label="CFG Scale", value=7.0)

                with gr.Row():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512)
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512)
                
                sampler = gr.Dropdown(
                    label='Sampling method',
                    choices=[s.name for s in shared.sd_samplers.samplers_for_img2img if "DDIM" not in s.name], # Filter samplers for better performance/compatibility
                    value='Euler a'
                )

            with gr.Column(scale=3):
                gr.HTML("<h4>Live Output</h4>")
                output_gallery = gr.Gallery(
                    label="Live Output", show_label=False, elem_id="sd_batch_stream_gallery"
                ).style(columns=[4], rows=[2], object_fit="contain", height="auto")

        # --- Event Handlers ---
        # This is the main click handler that starts the generation process.
        stream_button.click(
            fn=process_and_stream_images,
            inputs=[prompts_input, negative_prompt, steps, cfg_scale, width, height, sampler],
            outputs=[output_gallery, stream_button]
        )

        # Connect our interrupt/skip buttons to the web UI's built-in interruption logic.
        interrupt_button.click(fn=lambda: shared.state.interrupt(), inputs=None, outputs=None)
        skip_button.click(fn=lambda: shared.state.skip(), inputs=None, outputs=None)

        # Return the top-level Gradio Blocks container.
        return streamer_tab


# --- The Magic Hook ---
# This function is called by the web UI to build the tabs.
def add_new_tab_to_ui():
    # 1. Call our function to create the UI.
    new_tab = create_streamer_ui()
    
    # 2. Return a list of tuples. Each tuple is (gradio_block, "Tab Name", "unique_id").
    return [(new_tab, "Batch Streamer", "sd_batch_streamer_tab")]

# 3. Register our function to be called when the UI is being built.
on_ui_tabs(add_new_tab_to_ui)