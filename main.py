from fastapi import FastAPI, UploadFile
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline 
from vllm import LLM, SamplingParams
from pydantic import BaseModel
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import torch
import gc

app = FastAPI()


Get_The_Data = {"transcribe": "", "translate": "", "generate": "", "sentiment_analysis": "", "chat_model": ""}

class TranslationRequest(BaseModel):
    source_language: str
    target_language: str
    input_text: str
    
class promptmodel(BaseModel):
    prompt:str
    
class Translation(BaseModel):
    text: str
    target_lang: str

class sqlmodel(BaseModel):
    prompt : str
    schema : str

@app.get('/')
def index():
    return {'message': 'Welcome to our app. If you want to continue, go to the link http://127.0.0.1:8000/docs/'}

# Whisper Model
@app.post("/transcribe/")
async def transcribe_audio(Audio_file: UploadFile):
    whisper_model = "openai/whisper-small"
    asr_pipeline = pipeline("automatic-speech-recognition", model=whisper_model)
    audio_content = await Audio_file.read()
    transcription = asr_pipeline(audio_content)
    del whisper_model
    torch.cuda.empty_cache()
    Get_The_Data["transcribe"] = transcription
    return transcription

# Translate Model
@app.post("/translate/")
async def text_translate(request:Translation):
    trans_model = "facebook/m2m100_418M"
    pipe = pipeline("text2text-generation", model=trans_model)
    text = request.text
    target_lang = request.target_lang
    translated_text = pipe(text, forced_bos_token_id=pipe.tokenizer.get_lang_id(lang=target_lang))
    generated_text = translated_text[0]['generated_text']
    del trans_model
    torch.cuda.empty_cache()
    Get_The_Data["translate"] = generated_text
    return generated_text

# Text to Text Model
@app.post("/generate/")
async def generate_text(request: promptmodel):
    gen_model = "google/flan-t5-base"
    generator = pipeline("text2text-generation", model=gen_model)
    prompt= request.prompt
    generated_text = generator(prompt, max_length=100)
    gen_text = generated_text[0]['generated_text']
    del gen_model
    torch.cuda.empty_cache()
    Get_The_Data["generate"] = gen_text  
    return gen_text

# Model For SequenceClassification
@app.post("/sentiment_analysis/")
async def sentiment_analysis(request: promptmodel):
    auto_tokenizer = AutoTokenizer.from_pretrained("textattack/albert-base-v2-yelp-polarity")
    auto_model = AutoModelForSequenceClassification.from_pretrained("textattack/albert-base-v2-yelp-polarity")
    prompt= request.prompt
    inputs = auto_tokenizer(prompt, return_tensors="pt")
    outputs = auto_model(**inputs)
    sentiment = "positive" if outputs[0].argmax().item() == 1 else "negative"
    del auto_model
    torch.cuda.empty_cache()
    Get_The_Data["sentiment_analysis"] = sentiment
    return sentiment

# Chat Model
@app.post("/chat_model/")
async def chat_with_model(request: promptmodel):
    chat_model = 'TheBloke/toxicqa-Llama2-7B-AWQ'
    llm = LLM(model=chat_model, trust_remote_code=True, quantization="awq", dtype="half", gpu_memory_utilization=0.5)
    prompt = request.prompt
    prompt_template = f'''Below is an instruction that describes a task. Write a response that adequately completes the request.
       ### Instruction:
       {{prompt}}

       ### Response:
       '''
    prompt_with_template = prompt_template.format(prompt=prompt)
    outputs = llm.generate([prompt_with_template], (SamplingParams(temperature=0, max_tokens=200)))
    generated_text = outputs[0].outputs[0].text if outputs else ""
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    Get_The_Data["chat_model"] = generated_text
    return generated_text

#SQL Model
@app.post("/generate-sql/")
async def generate_sql(prompt_request:sqlmodel):
            sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
            prompt_template=f'''## Task
Generate a SQL query to answer the following question:
`{{prompt}}`

### Database Schema
{{schema}}

### SQL
Given the database schema, here is the SQL query that answers `{{prompt}}`:
```sql
'''
            llm = LLM(model="TheBloke/sqlcoder-7B-AWQ", quantization="awq", dtype="half",gpu_memory_utilization=0.7,max_model_len=1000)
            prompt = prompt_template.format(prompt=prompt_request.prompt,schema=prompt_request.schema)
            output = llm.generate([prompt], sampling_params)[0]
            generated_text = output.outputs[0].text
            destroy_model_parallel()
            del llm
            gc.collect()
            torch.cuda.empty_cache()
            torch.distributed.destroy_process_group()
            Get_The_Data["SQL_model"] = generated_text
            return generated_text

@app.get("/get_the_data/")
async def get_posted_text():
    return Get_The_Data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
