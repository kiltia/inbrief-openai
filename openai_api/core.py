import logging

import numpy as np
import openai
from tenacity import (
    after_log,
    before_log,
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

from openai_api.prompts import CLASSIFY_TASK, EXAMPLE_REQUEST, EXAMPLE_RESPONSE

logger = logging.getLogger(__name__)


@retry(
    wait=wait_exponential(min=2, max=60, multiplier=2),
    after=after_log(logger, logging.INFO),
    before=before_log(logger, log_level=logging.INFO),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)
async def aget_embeddings(input, model):
    embs = await openai.Embedding.acreate(input=input, model=model)["data"]

    return list(map(lambda x: x["embedding"], embs))


@retry(
    wait=wait_exponential(min=2, max=60, multiplier=2),
    after=after_log(logger, logging.INFO),
    before=before_log(logger, log_level=logging.INFO),
    before_sleep=before_sleep_log(logger, logging.INFO),
    reraise=True,
)
def get_embeddings(input, model):
    embs = openai.Embedding.create(input=input, model=model)["data"]

    return list(map(lambda x: x["embedding"], embs))


def summarize(input, model):
    return (
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Тебе необходимо суммаризовать текст в новый текст, сохранив только САМЫЕ основные моменты повествования. Необходимо уложиться в 300 символов.",
                },
                {"role": "user", "content": "\n".join(input)},
            ],
            temperature=0.2,
            presence_penalty=-1.5,
            timeout=30,
        )
    )["choices"][0]["message"]["content"]

async def asummarize(input, model):
    return (
        await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Тебе необходимо суммаризовать текст в новый текст, сохранив только САМЫЕ основные моменты повествования. Необходимо уложиться в 300 символов.",
                },
                {"role": "user", "content": "\n".join(input)},
            ],
            temperature=0.2,
            presence_penalty=-1.5,
            timeout=30,
        )
    )["choices"][0]["message"]["content"]

def get_title(input, model):
    return (
        openai.ChatCompletion.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Тебе будет дан набор текстов, принадлежащих к одному сюжету. Тебе будет необходимо выделить общие черты сюжета и придумать новостной заголовок для него. Отвечай только самим заголовком",
                },
                {"role": "user", "content": "\n".join(input)},
            ],
            temperature=0.2,
            presence_penalty=-1.5,
            timeout=30,
        )
    )["choices"][0]["message"]["content"]

async def aget_title(input, model):
    return (
        await openai.ChatCompletion.acreate(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Тебе будет дан набор текстов, принадлежащих к одному сюжету. Тебе будет необходимо выделить общие черты сюжета и придумать новостной заголовок для него. Отвечай только самим заголовком",
                },
                {"role": "user", "content": "\n".join(input)},
            ],
            temperature=0.2,
            presence_penalty=-1.5,
            timeout=30,
        )
    )["choices"][0]["message"]["content"]


def classify_attempt(attempt, categories, max_retries, **kwargs):
    def validate_response(response, categories):
        if response in categories:
            return (True, response)
        else:
            return (
                False,
                "Выводи только название одного из классов, перечисленных мной ранее,",
            )

    if attempt > max_retries:
        return None
    logging.info(f"Creating request {attempt + 1} for chat completion")
    completion = openai.ChatCompletion.create(**kwargs)
    response = completion["choices"][0]["message"]["content"]
    logging.info(f"Got response from OpenAI: {response}")
    status, value = validate_response(response, categories)
    if status:
        logging.info("Response has correct format")
        return value
    else:
        messages = kwargs.pop("messages")
        messages.append(
            {
                "role": "user",
                "content": value
                + "И напоминаю, не выводи ничего, кроме одного слова — названия класса.",
            }
        )
        logging.debug("Response to model: " + messages[-1]["content"])
        logging.warn("Got bad format from model. Trying another attempt")
        return classify_attempt(
            attempt + 1,
            categories,
            messages=messages,
            max_retries=max_retries,
            **kwargs,
        )


# TODO(nrydanov): Move stop_after_attempt to configuration file
# NOTE(nrydanov): This method make paid request to OpenAI
@retry(
    wait=wait_exponential(min=2, max=60, multiplier=2),
    after=after_log(logger, logging.INFO),
    before=before_log(logger, log_level=logging.INFO),
    before_sleep=before_sleep_log(logger, logging.INFO),
    stop=stop_after_attempt(3),
    reraise=True,
)
def classify(text, categories, model, max_retries):
    class_list = f"Вот список классов: {','.join(categories)}"
    max_tokens = np.max(list(map(lambda x: len(x), categories)))
    return classify_attempt(
        0,
        categories,
        model=model,
        messages=[
            {"role": "system", "content": CLASSIFY_TASK + "\n" + class_list},
            {"role": "user", "content": EXAMPLE_REQUEST},
            {"role": "assistant", "content": EXAMPLE_RESPONSE},
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        presence_penalty=-1.5,
        timeout=30,
        max_tokens=int(max_tokens),
        max_retries=max_retries,
    )
