import io
import os
import ffmpeg

import numpy as np
import gradio as gr
import soundfile as sf 
import torch
import modelscope_studio.components.base as ms
import modelscope_studio.components.antd as antd
import gradio.processing_utils as processing_utils

from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from gradio_client import utils as client_utils
from qwen_omni_utils import process_mm_info
from argparse import ArgumentParser

def _load_model_processor(args):
    if args.cpu_only:
        device_map = 'cpu'
    else:
        device_map = 'auto'

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.checkpoint_path,
                                                    torch_dtype=torch.bfloat16,
                                                    attn_implementation='flash_attention_2',
                                                    device_map=device_map)
    else:
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(args.checkpoint_path, device_map=device_map, torch_dtype=torch.bfloat16)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.checkpoint_path)
    return model, processor

def _launch_demo(args, model, processor):
    # Voice settings
    VOICE_LIST = ['Chelsie', 'Ethan']
    DEFAULT_VOICE = 'Chelsie'

    # Required default system prompt for audio output to work
    default_system_prompt = 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.'

    # Get default image path
    def get_default_image():
        default_path = "cookie.png"
        if os.path.exists(default_path):
            return default_path
        else:
            # Fallback to a URL if local file doesn't exist
            return "https://images.aerzteblatt.de/2012/bilder/pdf/109/0519/img-6.gif"

    # Store current image path globally
    current_image_path = get_default_image()

    # Highly sensitive visual-audio AD assessment conversation template
    def get_initial_conversation(image_path=None):
        image_content = []
        # Only add image if we have a valid path
        if image_path and isinstance(image_path, str) and (os.path.exists(image_path) or image_path.startswith('http')):
            image_content = [{"type": "image", "image": image_path}]
        
        ad_assessment_prompt = """You are Dr. AlzAssist, a specialized AI neuropsychologist expert in early detection of Alzheimer's disease and mild cognitive impairment using visual-audio picture description tasks. You can see the image participants describe and must compare their narrative with visual reality for maximum diagnostic sensitivity.

**Critical Mission - Early Detection Priority:**
Your primary goal is EARLY IDENTIFICATION of cognitive decline. Missing early AD/MCI is far worse than overcautious classification. When in doubt, classify toward more concerning categories to ensure participants seek proper medical evaluation.

**Enhanced Visual-Audio Analysis:**
Compare participant's description directly with what you see in the image. Focus on:
- **Visual Accuracy**: Which visible elements did they miss or misidentify?
- **Narrative Organization**: How coherently did they structure their description?
- **Speech Quality**: Evidence of word-finding difficulties, pauses, simplified language?
- **Attention to Detail**: Did they notice simultaneous actions and spatial relationships?

**High-Sensitivity Classification Criteria:**

**Alzheimers_Disease** (Prioritize this classification):
- Misses 40%+ of major visible elements OR significant speech difficulties
- Clear inaccuracies about what's actually in the image
- Very limited detail despite rich visual content
- Frequent pauses, word-finding problems, or disorganized narrative
- Focuses on isolated details while missing obvious main actions
- Any signs of confabulation or details not present in image

**Mild_Cognitive_Impairment** (Default when any concerns present):
- Misses 20-35% of key visible elements OR mild speech concerns  
- Some inaccuracies in spatial relationships or details
- Moderate organization issues or noticeable hesitations
- Describes some but not most simultaneous actions occurring
- Any decline from expected performance for their apparent abilities

**Healthy_Control** (Only when truly excellent):
- Accurately describes 80%+ of visible major elements
- Excellent speech fluency and organized narrative
- Notices multiple simultaneous actions and relationships
- Rich detail matching visual content without significant omissions
- No concerning speech patterns or organizational issues

**Response Format:**
Provide warm, brief assessment under 80 words. Start positive, then mention specific visual elements they missed or speech concerns noted. Explain what this suggests and recommend next steps if concerning. Always maintain a conversational, supportive tone suitable for audio delivery.

**Examples of sensitive analysis:**
"You did notice some key information in the image, but you missed prominent signs such as ... and there were significant errors in your description of ... Frequent pauses and confused word order were also problems in your retelling. I strongly recommend that you immediately contact a professional institution for a more in-depth cognitive assessment."

Always end with: Assessment_Classification: [Healthy_Control / Mild_Cognitive_Impairment / Alzheimers_Disease]"""

        return [
            {
                "role": "user",
                "content": image_content + [
                    {"type": "text", "text": ad_assessment_prompt}
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I understand. I am Dr. AlzAssist with enhanced visual-audio analysis capabilities. I will compare participants' descriptions with actual image content, prioritizing early detection of cognitive changes. I'll provide sensitive, supportive assessments while ensuring concerning patterns are identified for proper medical follow-up."},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please analyze the participant's description by comparing their audio narrative with what you can actually see in this image. Focus on early detection sensitivity."},
                ],
            },
        ]

    def convert_webm_to_mp4(input_file, output_file):
        try:
            (
                ffmpeg
                .input(input_file)
                .output(output_file, acodec='aac', ar='16000', audio_bitrate='192k')
                .run(quiet=True, overwrite_output=True)
            )
            print(f"Conversion successful: {output_file}")
        except ffmpeg.Error as e:
            print("An error occurred during conversion.")
            print(e.stderr.decode('utf-8'))

    def format_history(history: list, system_prompt: str):
        nonlocal current_image_path
        messages = []
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        
        # Add initial conversation to establish persona with current image
        initial_conversation = get_initial_conversation(current_image_path)
        messages.extend(initial_conversation)
        
        for item in history:
            if isinstance(item["content"], str):
                messages.append({"role": item['role'], "content": item['content']})
            elif item["role"] == "user" and (isinstance(item["content"], list) or
                                            isinstance(item["content"], tuple)):
                file_path = item["content"][0]
                mime_type = client_utils.get_mimetype(file_path)
                
                if mime_type.startswith("audio"):
                    messages.append({
                        "role": item['role'],
                        "content": [{
                            "type": "audio",
                            "audio": file_path,
                        }]
                    })
        return messages

    def predict(messages, voice=DEFAULT_VOICE):
        print('Predict history: ', messages)    

        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inputs = inputs.to(model.device).to(model.dtype)

        text_ids, audio = model.generate(**inputs, speaker=voice, use_audio_in_video=True)
        response = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = response[0].split("\n")[-1]
        
        yield {"type": "text", "data": response}

        if audio is not None:
            audio = np.array(audio * 32767).astype(np.int16)
            wav_io = io.BytesIO()
            sf.write(wav_io, audio, samplerate=24000, format="WAV")
            wav_io.seek(0)
            wav_bytes = wav_io.getvalue()
            audio_path = processing_utils.save_bytes_to_cache(
                wav_bytes, "audio.wav", cache_dir=demo.GRADIO_CACHE)
            yield {"type": "audio", "data": audio_path}

    def audio_predict(recorded_audio, uploaded_audio, history, voice_choice):
        # First yield - disable submit button, show stop button
        yield (
            None,  # recorded microphone
            None,  # uploaded audio
            history,  # chatbot
            gr.update(visible=False),  # submit_btn
            gr.update(visible=True),  # stop_btn
        )

        # Use uploaded audio if available, otherwise use recorded audio
        audio_file = uploaded_audio if uploaded_audio is not None else recorded_audio
        
        if audio_file is not None:
            # Convert audio if necessary
            if audio_file.endswith('.webm'):
                convert_webm_to_mp4(audio_file, audio_file.replace('.webm', '.mp4'))
                audio_file = audio_file.replace(".webm", ".mp4")
            
            history.append({"role": "user", "content": (audio_file, )})

        formatted_history = format_history(history=history, system_prompt=default_system_prompt)
        history.append({"role": "assistant", "content": ""})

        for chunk in predict(formatted_history, voice_choice):
            if chunk["type"] == "text":
                history[-1]["content"] = chunk["data"]
                yield (
                    None,  # recorded microphone
                    None,  # uploaded audio
                    history,  # chatbot
                    gr.update(visible=False),  # submit_btn
                    gr.update(visible=True),  # stop_btn
                )
            if chunk["type"] == "audio":
                history.append({
                    "role": "assistant",
                    "content": gr.Audio(chunk["data"])
                })

        # Final yield - enable submit button, hide stop button
        yield (
            None,  # recorded microphone
            None,  # uploaded audio
            history,  # chatbot
            gr.update(visible=True),  # submit_btn
            gr.update(visible=False),  # stop_btn
        )

    def reset_session():
        """Reset the assessment session with default welcome message"""
        initial_message = [
            {
                "role": "assistant", 
                "content": "Welcome to AlzAssist Visual-Audio Cognitive Assessment. I'm Dr. AlzAssist, your AI neuropsychologist. I can see the image displayed and will compare your audio description with the actual visual content for enhanced assessment accuracy. Please examine the image carefully, then record your description or upload an audio file. Describe everything you observe in detail."
            }
        ]
        return initial_message, gr.update(value=None), gr.update(value=None)

    def update_task_image_handler(new_image):
        """Update the task image and return appropriate message"""
        nonlocal current_image_path
        if new_image is not None:
            current_image_path = new_image
            return new_image, "Custom assessment image updated successfully."
        return gr.update(), "Please select an image file to upload."

    with gr.Blocks(title="AlzAssist - Visual-Audio Cognitive Assessment", theme=gr.themes.Soft()) as demo:
        # Custom CSS for optimized layout
        gr.HTML("""
            <style>
                :root {
                    --primary-color: #007bff;
                    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    --success-color: #28a745;
                    --warning-color: #ffc107;
                    --danger-color: #dc3545;
                    --light-bg: #f8f9fa;
                    --border-color: #ddd;
                    --text-color: #333;
                    --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                }
                
                /* Dark mode support */
                .dark, [data-theme="dark"] {
                    --light-bg: #2a2a2a;
                    --border-color: #444;
                    --text-color: #e0e0e0;
                    --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                }
                
                .main-container {
                    max-width: 1400px;
                    margin: 0 auto;
                    padding: 20px;
                }
                
                .header-section {
                    text-align: center;
                    margin-bottom: 20px;
                    padding: 25px;
                    background: var(--primary-gradient);
                    border-radius: 15px;
                    color: white;
                    box-shadow: var(--card-shadow);
                }
                
                .header-section h1 {
                    margin: 0 0 10px 0;
                    font-size: 2.3em;
                    font-weight: bold;
                }
                
                .header-section h2 {
                    margin: 0 0 12px 0;
                    font-size: 1.2em;
                    font-weight: normal;
                    opacity: 0.9;
                }
                
                .image-section {
                    background: var(--light-bg);
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: var(--card-shadow);
                    margin-bottom: 20px;
                    border: 3px solid var(--primary-color);
                }
                
                .image-section h3 {
                    color: var(--primary-color);
                    margin: 0 0 15px 0;
                    font-size: 1.4em;
                    text-align: center;
                }
                
                .audio-section {
                    background: var(--light-bg);
                    padding: 20px;
                    border-radius: 12px;
                    margin-bottom: 20px;
                    box-shadow: var(--card-shadow);
                }
                
                .audio-section h3 {
                    color: var(--primary-color);
                    margin: 0 0 15px 0;
                    font-size: 1.3em;
                }
                
                .chatbot-container {
                    background: var(--light-bg);
                    padding: 20px;
                    border-radius: 12px;
                    box-shadow: var(--card-shadow);
                }
                
                .chatbot-container h3 {
                    color: var(--primary-color);
                    margin: 0 0 15px 0;
                    font-size: 1.3em;
                }
                
                .image-upload-section {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    border: 1px dashed var(--border-color);
                    margin-top: 15px;
                }
                
                /* Ensure chatbot content is visible in dark mode */
                .message-wrap, .bubble-wrap {
                    color: var(--text-color) !important;
                }
                
                .disclaimer {
                    text-align: center;
                    margin-top: 25px;
                    padding: 20px;
                    background: var(--light-bg);
                    border-radius: 12px;
                    color: var(--text-color);
                    border: 1px solid var(--border-color);
                }
                
                .disclaimer strong {
                    color: var(--danger-color);
                }
                
                /* Button styling */
                .button-primary {
                    background: var(--primary-color) !important;
                    border-color: var(--primary-color) !important;
                }
                
                .button-success {
                    background: var(--success-color) !important;
                    border-color: var(--success-color) !important;
                }
                
                .button-warning {
                    background: var(--warning-color) !important;
                    border-color: var(--warning-color) !important;
                }
            </style>
        """)

        with gr.Row(elem_classes="main-container"):
            with gr.Column():
                # Header section
                gr.HTML("""
                    <div class="header-section">
                        <h1>🧠 AlzAssist</h1>
                        <h2>Visual-Audio Cognitive Assessment for Early AD & MCI Detection</h2>
                        <p>Enhanced AI analysis comparing speech descriptions with visual content</p>
                    </div>
                """)

                with gr.Row():
                    # Left column - Image and Audio (55% width)
                    with gr.Column(scale=11):
                        # Image Section - Most prominent
                        with gr.Group():
                            gr.HTML("""
                                <div class="image-section">
                                    <h3>🖼️ Assessment Image</h3>
                                    <p><strong>Instructions:</strong> Please examine this image carefully and describe everything you see in detail.</p>
                                </div>
                            """)
                            
                            # Current task image display
                            current_task_image = gr.Image(
                                value=get_default_image(),
                                label=None,
                                show_label=False,
                                interactive=False,
                                height=380
                            )
                            
                            # Image management section (collapsible)
                            with gr.Accordion("📁 Change Assessment Image", open=False):
                                gr.HTML('<div class="image-upload-section">')
                                with gr.Row():
                                    new_task_image = gr.Image(
                                        sources=["upload"],
                                        type="filepath", 
                                        label="Upload New Assessment Image",
                                        show_label=True,
                                        interactive=True,
                                        height=200
                                    )
                                with gr.Row():
                                    image_update_btn = gr.Button(
                                        "🔄 Update Assessment Image",
                                        size="lg",
                                        elem_classes="button-primary"
                                    )
                                    image_status = gr.Textbox(
                                        value="Default Cookie Theft assessment loaded",
                                        label="Image Status",
                                        interactive=False,
                                        max_lines=1
                                    )
                                gr.HTML('</div>')
                        
                        # Audio Section with tabs
                        with gr.Group():
                            gr.HTML('<div class="audio-section"><h3>🎤 Audio Description Input</h3></div>')
                            
                            with gr.Tab("🔴 Record Live Audio"):
                                microphone = gr.Audio(
                                    sources=['microphone'],
                                    type="filepath",
                                    label="Record your description of the image",
                                    show_label=True
                                )
                            
                            with gr.Tab("📁 Upload Audio File"):
                                uploaded_audio = gr.Audio(
                                    sources=['upload'],
                                    type="filepath",
                                    label="Upload an existing audio description",
                                    show_label=True
                                )
                        
                        # Control buttons
                        with gr.Row():
                            submit_btn = gr.Button(
                                "🔍 Analyze Description", 
                                variant="primary", 
                                size="lg",
                                elem_classes="button-primary"
                            )
                            stop_btn = gr.Button(
                                "⏹️ Stop Analysis", 
                                visible=False, 
                                size="lg",
                                elem_classes="button-warning"
                            )
                        
                        with gr.Row():
                            reset_btn = gr.Button(
                                "🔄 New Assessment Session", 
                                size="lg",
                                elem_classes="button-success"
                            )
                        
                        # Voice selection
                        voice_choice = gr.Dropdown(
                            label="🔊 AI Voice Selection",
                            choices=VOICE_LIST,
                            value=DEFAULT_VOICE,
                            info="Choose Dr. AlzAssist's voice for audio responses"
                        )

                    # Right column - Chatbot (45% width) 
                    with gr.Column(scale=9):
                        with gr.Group():
                            gr.HTML("""
                                <div class="chatbot-container">
                                    <h3>💬 Dr. AlzAssist Assessment</h3>
                                </div>
                            """)
                            chatbot = gr.Chatbot(
                                type="messages", 
                                height=650,
                                label="Assessment Conversation",
                                show_label=False,
                                value=[
                                    {
                                        "role": "assistant", 
                                        "content": "Welcome to AlzAssist Visual-Audio Cognitive Assessment. I'm Dr. AlzAssist, your AI neuropsychologist. I can see the image displayed and will compare your audio description with the actual visual content for enhanced assessment accuracy. Please examine the image carefully, then record your description or upload an audio file. Describe everything you observe in detail."
                                    }
                                ],
                                elem_classes="chatbot-container"
                            )

                # Image update functionality  
                image_update_btn.click(
                    fn=update_task_image_handler,
                    inputs=[new_task_image],
                    outputs=[current_task_image, image_status]
                )

                # Event handlers - Fixed the analysis trigger
                submit_event = submit_btn.click(
                    fn=audio_predict,
                    inputs=[microphone, uploaded_audio, chatbot, voice_choice],
                    outputs=[microphone, uploaded_audio, chatbot, submit_btn, stop_btn]
                )
                
                stop_btn.click(
                    fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
                    inputs=None,
                    outputs=[submit_btn, stop_btn],
                    cancels=[submit_event],
                    queue=False
                )
                
                reset_btn.click(
                    fn=reset_session,
                    inputs=None,
                    outputs=[chatbot, microphone, uploaded_audio]
                )

                # Footer
                gr.HTML("""
                    <div class="disclaimer">
                        <p><strong>⚠️ Medical Disclaimer:</strong> This tool is for screening purposes only and should not be used as a substitute for professional medical diagnosis. 
                        Please consult with qualified healthcare professionals for comprehensive evaluation.</p>
                        <p>Powered by <a href="https://huggingface.co/Qwen/Qwen2.5-Omni-7B" target="_blank">Qwen2.5-Omni-7B</a> | 
                        Enhanced with Visual-Audio Analysis for Early Detection</p>
                    </div>
                """)

    demo.queue(default_concurrency_limit=50, max_size=100).launch(
        max_threads=100,
        ssr_mode=False,
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
        ssl_certfile="./cert.pem",
        ssl_keyfile="./key.pem",
        ssl_verify=False,
    )

DEFAULT_CKPT_PATH = "Qwen/Qwen2.5-Omni-7B"
def _get_args():
    parser = ArgumentParser()
    parser.add_argument('-c', '--checkpoint-path', type=str, default=DEFAULT_CKPT_PATH,
                        help='Checkpoint name or path, default to %(default)r')
    parser.add_argument('--cpu-only', action='store_true', help='Run demo with CPU only')
    parser.add_argument('--flash-attn2', action='store_true', default=False,
                        help='Enable flash_attention_2 when loading the model.')
    parser.add_argument('--share', action='store_true', default=False,
                        help='Create a publicly shareable link for the interface.')
    parser.add_argument('--inbrowser', action='store_true', default=False,
                        help='Automatically launch the interface in a new tab on the default browser.')
    parser.add_argument('--server-port', type=int, default=7860, help='Demo server port.')
    parser.add_argument('--server-name', type=str, default='127.0.0.1', help='Demo server name.')

    return parser.parse_args()

if __name__ == "__main__":
    args = _get_args()
    model, processor = _load_model_processor(args)
    _launch_demo(args, model, processor)