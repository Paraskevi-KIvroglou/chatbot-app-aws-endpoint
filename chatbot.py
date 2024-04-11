import json
import boto3
from typing import Dict
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
import langchain.globals

from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler

class HFContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        self.len_prompt = len(prompt)
        input_dict = {
            "inputs": prompt,
            "parameters": model_kwargs
        }
        input_str = json.dumps(input_dict)
        print(input_str)
        return input_str.encode('utf-8')

    def transform_output(self, output: bytes) -> str:
        response_json = output.read()
        res = json.loads(response_json)
        print(res)

        # stripping away the input prompt from the returned response
        ans = res[0]['generated_text'][self.len_prompt:]
        ans = ans[:ans.rfind("Human")].strip()
        return ans

parameters = {
    'do_sample':True,
    'top_p': 0.8,
    'max_new_tokens': 512,
    'temperature':0.4,
    'watermark':True       
}
#Parameters
AWS_ENDPOINT = "huggingface-pytorch-tgi-inference-2024-04-09-22-57-27-755"
AWS_REGION = "us-east-1"

#Create a runtime client
roleARN = "arn:aws:iam::381492103890:role/sagemaker-access"
sts_client = boto3.client("sts")
response = sts_client.assume_role(
    RoleArn=roleARN, RoleSessionName="CrossAccountSession"
)

client = boto3.client(
    "sagemaker-runtime",
    region_name="us-east-1",
    aws_access_key_id=response["Credentials"]["AccessKeyId"],
    aws_secret_access_key=response["Credentials"]["SecretAccessKey"],
    aws_session_token=response["Credentials"]["SessionToken"]
)

def load_model():
    llm = SagemakerEndpoint(
        endpoint_name=f"{AWS_ENDPOINT}",
        client = client,
        model_kwargs=parameters,
        content_handler=HFContentHandler(),
    )
    return llm

def demo_miny_memory():
    llm_data = load_model()
    memory = ConversationBufferMemory(llm = llm_data,max_token_limit = 512)
    return memory

def demo_chain(input_text, memory):
    llm_data = load_model()
    llm_conversation = ConversationChain(llm=llm_data,memory=memory,verbose=langchain.globals.get_verbose())

    chat_reply = llm_conversation.predict(input=input_text)
    return chat_reply