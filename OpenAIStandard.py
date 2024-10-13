import io
import base64
import time
import uuid
import torch
from fastapi import FastAPI
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union

# Define models for the request body
class ImageContent(BaseModel):
    type: str = Field(..., description="Type of the content, e.g., 'text' or 'image_url'")
    text: Optional[str] = Field(None, description="Text content if type is 'text'")
    image_url: Optional[Dict[str, str]] = Field(None, description="Base64-encoded image URL if type is 'image_url'")

class Message(BaseModel):
    role: str = Field(..., description="The role of the sender, e.g., 'user'")
    content: Union[str, List[ImageContent]] = Field(..., description="Content can be a text or a list of ImageContent")

    @validator('content', pre=True)
    def validate_content(cls, value):
        if isinstance(value, str):
            return value
        elif isinstance(value, list):
            return [ImageContent(**item) for item in value]
        raise ValueError("Content must be either a string or a list of ImageContent")
    
class ChatRequest(BaseModel):
    model: str = Field(..., description="The model name to be used for completion")
    messages: List[Message] = Field(..., description="List of chat messages")
    temperature: float = Field(0.7, description="Temperature for sampling")
    max_tokens: int = Field(100, description="Maximum number of tokens to generate")
    top_p: float = Field(1.0, description="Top-p sampling for nucleus sampling")
    frequency_penalty: float = Field(0.0, description="Penalty for frequent tokens")
    presence_penalty: float = Field(0.0, description="Penalty for new token presence")

# Initialize FastAPI with custom settings
app = FastAPI(
    title="Chat Completion API",
    description="An API that supports text and image-based chat completion using OpenAI standards.",
    version="1.0.0",
    contact={
        "name": "Your Name",
        "email": "your.email@example.com",
    }
)

model_path = "./"
kwargs = {}
kwargs['torch_dtype'] = torch.bfloat16

processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2').cuda()
 
user_prompt = '<|user|>\n'
assistant_prompt = '<|assistant|>\n'
prompt_suffix = "<|end|>\n"

@app.post("/chat/completions", tags=["Chat Completion"])
async def chat_completion(request: ChatRequest):
    """
    Generates a chat completion response based on the provided prompt and optional image URLs.
    """
    messages = request.messages
    max_tokens = request.max_tokens
    
    prompt = ""
    images = []
    image_count = 0  # Keep track of image count for dynamic tagging

    # Process each message and build the prompt
    for message in messages:
        if message.role == "user":
            prompt += user_prompt
        elif message.role == "assistant":
            prompt += assistant_prompt
        
        # Add the content of the message
        if isinstance(message.content, str):
            prompt += f"{message.content}"
        else:
            for content in message.content:
                if content.type == "text":
                    prompt += f"{content.text}"
                elif content.type == "image_url" and content.image_url:
                    image_count += 1
                    image_data = content.image_url['url'].split(",")[1]
                    image_bytes = base64.b64decode(image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    images.append(image)
                    prompt += f"<|image_{image_count}|>"

        prompt += prompt_suffix

    prompt += assistant_prompt

    # Process inputs with text
    inputs = processor(prompt, images=images, return_tensors="pt").to("cuda:0")  # Pass images to the processor
    generate_ids = model.generate(**inputs, max_new_tokens=max_tokens, eos_token_id=processor.tokenizer.eos_token_id)
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    processorResponse = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Simulate token usage for demo purposes
    prompt_tokens = len(prompt.split())
    completion_tokens = len(processorResponse.split())
    total_tokens = prompt_tokens + completion_tokens

    response = {
        "id": f"chatcmpl-{str(uuid.uuid4())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "phi-3.5-visoin",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": processorResponse,
                    "tool_calls": [],
                    "refusal": None,
                    "function_call": None
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [],
                    "refusal": []
                }
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=81)

# uvicorn OpenAIStandard:app --host 127.0.0.1 --port 81 --reload
