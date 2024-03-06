from xai_components.base import InArg, OutArg, InCompArg, Component, BaseComponent, secret, xai_component
import boto3
import json

@xai_component
class BedrockAuthorize(Component):
    """Sets up the AWS Bedrock client.

    ##### inPorts:
    - aws_access_key_id: AWS Access Key ID.
    - aws_secret_access_key: AWS Secret Access Key.
    - aws_session_token: AWS Session Token (optional, use for temporary credentials).
    - region_name: AWS region name.
    """
    aws_access_key_id: InArg[secret]
    aws_secret_access_key: InArg[secret]
    aws_session_token: InArg[secret]
    region_name: InArg[str]

    def execute(self, ctx) -> None:
        boto3.setup_default_session(
            aws_access_key_id=self.aws_access_key_id.value,
            aws_secret_access_key=self.aws_secret_access_key.value,
            aws_session_token=self.aws_session_token.value,
            region_name=self.region_name.value
        )
        ctx['bedrock_client'] = boto3.client(service_name='bedrock-runtime')


def encode_prompt(model_id: str, conversation: list):
    ret = ''

    if model_id.startswith('anthropic.claude-v2') or model_id.startswith('anthropic.claude-instant-v1'):
        for message in conversation:
            if message['role'] == 'system':
                ret += message['content']
            elif message['role'] == 'user':
                ret += '\n\nHuman: ' + message['content']
            else:
                ret += '\n\nAssistant:' + message['content']
        ret += '\n\nAssistant:'
    elif model_id.startswith('anthropic.claude-3'): #Special snowflake case.
        ret_messages = []
        for message in conversation:
            ret_messages.append({
                'role': message['role'],
                'content': [
                    {
                        'type': 'text',
                        'text': message['content']
                    }
                ]
            })
        ret = ret_messages

    elif model_id.startswith('amazon.titan'):
        message_text = ''
        for message in conversation:
            if message['role'] == 'user':
                message_text += "User: " + message['content'] + "\n\n"
            elif message['role'] == 'system':
                message_text += "System: " + message['content'] + "\n\n"
            elif message['role'] == 'assistant':
                message_text += "Bot: " + message['content'] + "\n\n"
        message_text += 'Bot:'

        ret = message_text
    else:
        # Encode as if OpenAI:
        for message in conversation:
            ret += message['role'] + "|> " + message['content'] + "\n\n"
        ret += 'assistant|>'

    return ret


@xai_component
class BedrockInvokeModelChat(Component):
    """Invokes a model on AWS Bedrock in chat style and returns the response.

    ##### inPorts:
    - model_id: The model ID to invoke. (Example: ai21.j2-ultra-v1)
    - system_prompt: An (optional) instructional prompt to prepend to the conversation.
    - user_prompt: The prompt string to send to the model (appending to the conversation if present)
    - conversation: Optional previous chat conversation.
    - max_tokens: The maximum number of tokens to sample.
    - temperature: Sampling temperature.
    - top_k: Optional: Limit sample to the k most likely next tokens.
    - top_p: Optional: Cumulative probability cutoff for token selection.

    ##### outPorts:
    - completion: The completion response from the model.
    - out_conversation: The conversation after adding the model response.

    """
    model_id: InArg[str]
    system_prompt: InArg[str]
    user_prompt: InArg[str]
    conversation: InArg[list]
    max_tokens: InCompArg[int]
    temperature: InCompArg[float]
    top_k: InArg[int]
    top_p: InArg[float]

    completion: OutArg[str]
    out_conversation: OutArg[list]

    def execute(self, ctx) -> None:
        bedrock_client = ctx.get('bedrock_client')
        if bedrock_client is None:
            raise Exception("Bedrock client has not been authorized")

        conversation = self.conversation.value if self.conversation.value is not None else []
        if self.system_prompt.value is not None:
            conversation = [{'role': 'system', 'content': self.system_prompt.value}] + conversation
        if self.user_prompt.value is not None:
            conversation = conversation + [{'role': 'user', 'content': self.user_prompt.value}]


        # Add additional parameters based on the modelId
        if self.model_id.value.startswith("anthropic.claude-3"):
            messages = encode_prompt(self.model_id.value, conversation)

            body_data = {
                "messages": messages,
                "max_tokens": self.max_tokens.value,
                "anthropic_version": "bedrock-2023-05-31"
            }
        elif self.model_id.value.startswith("anthropic."):
            prompt = encode_prompt(self.model_id.value, conversation)

            body_data = {
                "prompt": prompt,
                "max_tokens_to_sample": self.max_tokens.value,
                "top_k": self.top_k.value if self.top_k.value is not None else 250, # Need a good default.
                "top_p": self.top_p.value if self.top_p.value is not None else 1,
                "temperature": self.temperature.value,
                "stop_sequences": ["\n\nHuman:"],
                "anthropic_version": "bedrock-2023-05-31"
            }
        elif self.model_id.value.startswith("cohere."):
            prompt = encode_prompt(self.model_id.value, conversation)

            body_data = {
                "prompt": prompt,
                "max_tokens": self.max_tokens.value,
                "temperature": self.temperature.value,
                "p": self.top_p.value if self.top_p.value is not None else 1,
                "k": self.top_k.value if self.top_k.value is not None else 0, #0?
            }

        elif self.model_id.value.startswith("meta."):
            prompt = encode_prompt(self.model_id.value, conversation)

            body_data = {
                "prompt": prompt,
                "max_gen_len": self.max_tokens.value,
                "temperature": self.temperature.value,
                "top_p": self.top_p.value if self.top_p.value is not None else 0.9
            }
        elif self.model_id.value.startswith("ai21."):
            prompt = encode_prompt(self.model_id.value, conversation)
            body_data = {
                "prompt": prompt,
                "temperature": self.temperature.value,
                "maxTokens": self.max_tokens.value,
                "topP": self.top_p.value if self.top_p.value is not None else 0.9,
                "stopSequences": ["\n\nuser|>"],
                "countPenalty": {"scale": 0},
                "presencePenalty": {"scale": 0},
                "frequencyPenalty": {"scale": 0}
            }
        else:
            prompt = encode_prompt(self.model_id.value, conversation)

            body_data = {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens.value,
                    "stopSequences": ["User:"],
                    "temperature": self.temperature.value,
                    "topP": self.top_p.value if self.top_p.value is not None else 1
                }
            }


        body = json.dumps(body_data)

        response = bedrock_client.invoke_model(
            body=body,
            modelId=self.model_id.value,
            accept='application/json',
            contentType='application/json'
        )        

        response_body = json.loads(response.get('body').read())

        if self.model_id.value.startswith('amazon.titan'):
            text = response_body.get('results')[0].get('outputText')
        else:
            text = response_body.get('completions')[0].get('data').get('text')

        self.out_conversation.value = []
        for message in conversation:
            self.out_conversation.value.append(message)

        self.out_conversation.value.append({'role': 'assistant', 'content': text})
        self.completion.value = text



@xai_component
class BedrockInvokeModel(Component):
    """Invokes a text completion model on AWS Bedrock and returns the response.

    ##### inPorts:
    - model_id: The model ID to invoke. (Example: ai21.j2-ultra-v1)
    - prompt: The prompt string to send to the model (appending to the conversation if present)
    - max_tokens: The maximum number of tokens to sample.
    - temperature: Sampling temperature.
    - top_k: Optional: Limit sample to the k most likely next tokens.
    - top_p: Optional: Cumulative probability cutoff for token selection.

    ##### outPorts:
    - completion: The completion response from the model.
    """

    model_id: InArg[str]
    prompt: InCompArg[str]
    max_tokens: InCompArg[int]
    temperature: InCompArg[float]
    top_k: InArg[int]
    top_p: InArg[float]

    completion: OutArg[str]

    def execute(self, ctx) -> None:
        bedrock_client = ctx.get('bedrock_client')
        if bedrock_client is None:
            raise Exception("Bedrock client has not been authorized")

        if self.model_id.value.startswith("anthropic."):


            body_data = {
                "prompt": self.prompt.value,
                "max_tokens_to_sample": self.max_tokens.value,
                "top_k": self.top_k.value if self.top_k.value is not None else 250, # Need a good default.
                "top_p": self.top_p.value if self.top_p.value is not None else 1,
                "temperature": self.temperature.value,
                "stop_sequences": [""],
                "anthropic_version": "bedrock-2023-05-31"
            }
        elif self.model_id.value.startswith("cohere."):
            body_data = {
                "prompt": self.prompt.value,
                "max_tokens": self.max_tokens.value,
                "temperature": self.temperature.value,
                "p": self.top_p.value if self.top_p.value is not None else 1,
                "k": self.top_k.value if self.top_k.value is not None else 0, #0?
            }

        elif self.model_id.value.startswith("meta."):
            body_data = {
                "prompt": self.prompt.value,
                "max_gen_len": self.max_tokens.value,
                "temperature": self.temperature.value,
                "top_p": self.top_p.value if self.top_p.value is not None else 0.9
            }
        elif self.model_id.value.startswith("ai21."):
            body_data = {
                "prompt": self.prompt.value,
                "temperature": self.temperature.value,
                "maxTokens": self.max_tokens.value,
                "topP": self.top_p.value if self.top_p.value is not None else 0.9,
                "stopSequences": [""],
                "countPenalty": {"scale": 0},
                "presencePenalty": {"scale": 0},
                "frequencyPenalty": {"scale": 0}
            }
        else:

            body_data = {
                "inputText": self.prompt.value,
                "textGenerationConfig": {
                    "maxTokenCount": self.max_tokens.value,
                    "stopSequences": [],
                    "temperature": self.temperature.value,
                    "topP": self.top_p.value if self.top_p.value is not None else 1
                }
            }


        body = json.dumps(body_data)

        response = bedrock_client.invoke_model(
            body=body,
            modelId=self.model_id.value,
            accept='application/json',
            contentType='application/json'
        )        

        response_body = json.loads(response.get('body').read())

        if self.model_id.value.startswith('amazon.titan'):
            text = response_body.get('results')[0].get('outputText')
        else:
            text = response_body.get('completions')[0].get('data').get('text')

        self.completion.value = text
        