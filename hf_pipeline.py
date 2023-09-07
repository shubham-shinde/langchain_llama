from llama2_load import tokenizer, device, model
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList
import torch

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [
    torch.LongTensor(tokenizer(x)['input_ids']).to(device)
    for x in stop_list
]

class StopOnTokens(StoppingCriteria):
    def __call__(
            self,
            input_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs
        ) -> bool:

        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])


generate_text = transformers.pipeline(
    model = model,
    tokenizer=tokenizer,
    return_full_text = True,
    task='text-generation',
    stopping_criteria = stopping_criteria,
    temperature = 0.1,
    max_new_tokens = 512,
    repetition_penalty = 1.1
)

if __name__ == '__main__':
    res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse")
    print(res[0]['generated_text'])


