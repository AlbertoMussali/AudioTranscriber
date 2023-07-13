# Serves as the main starting point for the Gradio-based UI of the transcription tool

import gradio as gr

with gr.Blocks() as my_demo:
    gr.Markdown("# Audio Transcription Tool!")
    gr.Markdown("## Powered by OpenAI's Whisper")
    inp_modelType = gr.Radio(choices=['Tiny', 'Base', 'Small', 'Medium', 'Large'], value='Medium', type='value', label='Model Size', info='Tiny/Base are faster but less accurate, Medium/Large are slower but more accurate. Use larger models for translation tasks.', interactive=True)
    gr.Markdown('## Inputs')
    inp_inputFiles = gr.File(file_count="multiple", label="Audio files to use as input.")
    inp_outputFolder = gr.File(file_count='directory', label='Folder to store all outputs.')
    
    
    
    # out = gr.Textbox(inp_modelType.value + "\n" + inp_inputFiles.value + "\n" + inp_outputFolder.value)


if __name__ == "__main__":
    my_demo.launch()