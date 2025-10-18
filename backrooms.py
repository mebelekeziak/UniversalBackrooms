import anthropic
from openai import OpenAI
import json
import datetime
import os
import argparse
import dotenv
import sys
import colorsys
import requests
from dataclasses import dataclass
from typing import List, Optional

# Attempt to load from .env file, but don't override existing env vars
dotenv.load_dotenv(override=False)

MODEL_INFO = {
    "sonnet": {
        "api_name": "claude-3-5-sonnet-20240620",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "opus": {
        "api_name": "claude-3-opus-20240229",
        "display_name": "Claude",
        "company": "anthropic",
    },
    "gpt4o": {
        "api_name": "gpt-4o",
        "display_name": "GPT-4o",
        "company": "openai",
        "is_reasoning": False,
    },
    "gpt5": {
        "api_name": "gpt-5",
        "display_name": "GPT-5",
        "company": "openai",
        "is_reasoning": True,
    },
    "gpt5-mini": {
        "api_name": "gpt-5-mini",
        "display_name": "GP5-T-5 Mini",
        "company": "openai",
        "is_reasoning": True,
    },
}

REASONING_MODELS = {
    info["api_name"]
    for info in MODEL_INFO.values()
    if info.get("company") == "openai" and info.get("is_reasoning", False)
}
USE_OPENAI_RESPONSES = True

LOG_FILE_ENCODING = "utf-8"
LOG_FILE_ERROR_MODE = "replace"


@dataclass
class ModelReply:
    text: str
    response_id: Optional[str] = None
    reasoning_item_ids: Optional[List[str]] = None
    encrypted_reasoning: Optional[List[str]] = None
    reasoning_summary: Optional[str] = None


def _normalize_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(item) for item in content)
    return str(content)


def _content_block_for_role(role: str, text: str) -> dict:
    content_type = "output_text" if role == "assistant" else "input_text"
    return {"type": content_type, "text": text}


def claude_conversation(actor, model, context, system_prompt=None):
    messages = [{"role": m["role"], "content": m["content"]} for m in context]

    # If Claude is the first model in the conversation, it must have a user message
    kwargs = {
        "model": model,
        "max_tokens": 1024,
        "temperature": 1.0,
        "messages": messages,
    }
    if system_prompt:
        kwargs["system"] = system_prompt
    message = anthropic_client.messages.create(**kwargs)
    content = message.content[0].text if message.content else ""
    return ModelReply(text=content)


def openai_conversation(
    actor,
    model,
    context,
    system_prompt=None,
    *,
    include_encrypted_reasoning=False,
    reasoning_effort=None,
    max_output_tokens=1024,
):
    input_items = []

    if system_prompt:
        input_items.append(
            {
                "role": "system",
                "content": [_content_block_for_role("system", system_prompt)],
            }
        )

    for message in context:
        normalized_content = _normalize_content(message["content"])
        input_items.append(
            {
                "role": message["role"],
                "content": [
                    _content_block_for_role(message["role"], normalized_content)
                ],
            }
        )

    if not input_items:
        input_items.append(
            {
                "role": "user",
                "content": [_content_block_for_role("user", "")],
            }
        )

    create_params = {
        "model": model,
        "input": input_items,
        "temperature": 1.0,
        "max_output_tokens": max_output_tokens,
    }

    include_items = []
    is_reasoning_model = model in REASONING_MODELS
    reasoning_settings = {}

    if include_encrypted_reasoning and is_reasoning_model:
        include_items.append("reasoning.encrypted_content")
        create_params["store"] = True

    if reasoning_effort and is_reasoning_model:
        reasoning_settings["effort"] = reasoning_effort

    if is_reasoning_model:
        reasoning_settings["summary"] = "auto"

    if reasoning_settings:
        create_params["reasoning"] = reasoning_settings

    if include_items:
        create_params["include"] = include_items

    response = openai_client.responses.create(**create_params)
    text = (response.output_text or "").strip()

    reasoning_ids: List[str] = []
    encrypted_reasoning: List[str] = []
    reasoning_summary: Optional[str] = None

    for item in getattr(response, "output", []):
        if getattr(item, "type", None) == "reasoning":
            reasoning_ids.append(item.id)
            encrypted_blob = getattr(item, "encrypted_content", None)
            if encrypted_blob:
                encrypted_reasoning.append(encrypted_blob)

    reasoning_metadata = getattr(response, "reasoning", None)
    if reasoning_metadata:
        if isinstance(reasoning_metadata, dict):
            reasoning_summary = reasoning_metadata.get("summary")
        else:
            reasoning_summary = getattr(reasoning_metadata, "summary", None)

    if isinstance(reasoning_summary, str):
        reasoning_summary = reasoning_summary.strip() or None

    return ModelReply(
        text=text,
        response_id=response.id,
        reasoning_item_ids=reasoning_ids or None,
        encrypted_reasoning=encrypted_reasoning or None,
        reasoning_summary=reasoning_summary,
    )


def load_template(template_name, models):
    try:
        with open(
            f"templates/{template_name}.jsonl",
            "r",
            encoding=LOG_FILE_ENCODING,
            errors=LOG_FILE_ERROR_MODE,
        ) as f:
            configs = [json.loads(line) for line in f]

        companies = []
        actors = []
        for i, model in enumerate(models):
            if model.lower() == "cli":
                companies.append("CLI")
                actors.append("CLI")
            else:
                companies.append(MODEL_INFO[model]["company"])
                actors.append(f"{MODEL_INFO[model]['display_name']} {i+1}")

        for i, config in enumerate(configs):
            if models[i].lower() == "cli":
                config["cli"] = True
                continue

            config["system_prompt"] = config["system_prompt"].format(
                **{f"lm{j+1}_company": companies[j] for j in range(len(companies))},
                **{f"lm{j+1}_actor": actors[j] for j in range(len(actors))},
            )
            for message in config["context"]:
                message["content"] = message["content"].format(
                    **{f"lm{j+1}_company": companies[j] for j in range(len(companies))},
                    **{f"lm{j+1}_actor": actors[j] for j in range(len(actors))},
                )

            if (
                not USE_OPENAI_RESPONSES
                and models[i] in MODEL_INFO
                and MODEL_INFO[models[i]]["company"] == "openai"
                and config["system_prompt"]
            ):
                system_prompt_added = False
                for message in config["context"]:
                    if message["role"] == "user":
                        message["content"] = (
                            f"<SYSTEM>{config['system_prompt']}</SYSTEM>\n\n{message['content']}"
                        )
                        system_prompt_added = True
                        break
                if not system_prompt_added:
                    config["context"].append(
                        {
                            "role": "user",
                            "content": f"<SYSTEM>{config['system_prompt']}</SYSTEM>",
                        }
                    )
            config["cli"] = config.get("cli", False)
        return configs
    except FileNotFoundError:
        print(f"Error: Template '{template_name}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in template '{template_name}'.")
        exit(1)


def get_available_templates():
    template_dir = "./templates"
    templates = []
    for file in os.listdir(template_dir):
        if file.endswith(".jsonl"):
            templates.append(os.path.splitext(file)[0])
    return templates


def main():
    global anthropic_client
    global openai_client
    parser = argparse.ArgumentParser(
        description="Run conversation between two or more AI language models."
    )
    parser.add_argument(
        "--lm",
        choices=["sonnet", "opus", "gpt4o", "gpt5", "gpt5-mini", "cli"],
        nargs="+",
        default=["opus", "opus"],
        help="Choose the models for LMs or 'cli' for the world interface (default: opus opus)",
    )

    available_templates = get_available_templates()
    parser.add_argument(
        "--template",
        choices=available_templates,
        default="cli" if "cli" in available_templates else available_templates[0],
        help=f"Choose a conversation template (available: {', '.join(available_templates)})",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=float("10"),
        help="Maximum number of turns in the conversation (default: infinity)",
    )
    parser.add_argument(
        "--include-encrypted-reasoning",
        action="store_true",
        help="Request encrypted reasoning traces for GPT-5 family models (Responses API only).",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high"],
        default="medium",
        help="Set the reasoning effort hint for GPT-5 family models (default: medium).",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=14048,
        help="Maximum tokens to request from OpenAI models via the Responses API (default: 2048).",
    )
    args = parser.parse_args()

    models = args.lm
    lm_models = []
    lm_display_names = []

    companies = []
    actors = []

    for i, model in enumerate(models):
        if model.lower() == "cli":
            lm_display_names.append("CLI")
            lm_models.append("cli")
            companies.append("CLI")
            actors.append("CLI")
        else:
            if model in MODEL_INFO:
                lm_display_names.append(f"{MODEL_INFO[model]['display_name']} {i+1}")
                lm_models.append(MODEL_INFO[model]["api_name"])
                companies.append(MODEL_INFO[model]["company"])
                actors.append(f"{MODEL_INFO[model]['display_name']} {i+1}")
            else:
                print(f"Error: Model '{model}' not found in MODEL_INFO.")
                sys.exit(1)

    # Filter out models not in MODEL_INFO (like 'cli')
    anthropic_models = [
        model
        for model in models
        if model in MODEL_INFO and MODEL_INFO[model]["company"] == "anthropic"
    ]
    if anthropic_models:
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            print(
                "Error: ANTHROPIC_API_KEY must be set in the environment or in a .env file."
            )
            sys.exit(1)
        anthropic_client = anthropic.Client(api_key=anthropic_api_key)

    openai_models = [
        model
        for model in models
        if model in MODEL_INFO and MODEL_INFO[model]["company"] == "openai"
    ]
    if openai_models:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print(
                "Error: OPENAI_API_KEY must be set in the environment or in a .env file."
            )
            sys.exit(1)
        openai_client = OpenAI(api_key=openai_api_key)

    configs = load_template(args.template, models)

    openai_settings = {
        "include_encrypted_reasoning": args.include_encrypted_reasoning,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.openai_max_output_tokens,
    }

    assert len(models) == len(
        configs
    ), f"Number of LMs ({len(models)}) does not match the number of elements in the template ({len(configs)})"

    system_prompts = [config.get("system_prompt", "") for config in configs]
    contexts = [config.get("context", []) for config in configs]
    logs_folder = "BackroomsLogs"
    os.makedirs(logs_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{logs_folder}/{'_'.join(models)}_{args.template}_{timestamp}.txt"

    turn = 0
    while turn < args.max_turns:
        for i in range(len(models)):
            if models[i].lower() == "cli":
                lm_response = cli_conversation(contexts[i])
            else:
                lm_response = generate_model_response(
                    lm_models[i],
                    lm_display_names[i],
                    contexts[i],
                    system_prompts[i],
                    openai_settings=openai_settings,
                )
            process_and_log_response(
                lm_response,
                lm_display_names[i],
                filename,
                contexts,
                i,
            )
        turn += 1

    print(f"\nReached maximum number of turns ({args.max_turns}). Conversation ended.")
    append_to_log(
        filename,
        f"\nReached maximum number of turns ({args.max_turns}). Conversation ended.\n",
    )


def generate_model_response(model, actor, context, system_prompt, openai_settings=None):
    if model.startswith("claude-"):
        return claude_conversation(
            actor, model, context, system_prompt if system_prompt else None
        )
    else:
        settings = openai_settings or {}
        return openai_conversation(
            actor,
            model,
            context,
            system_prompt if system_prompt else None,
            include_encrypted_reasoning=settings.get(
                "include_encrypted_reasoning", False
            ),
            reasoning_effort=settings.get("reasoning_effort"),
            max_output_tokens=settings.get("max_output_tokens", 2048),
        )


def generate_distinct_colors():
    hue = 0
    golden_ratio_conjugate = 0.618033988749895
    while True:
        hue += golden_ratio_conjugate
        hue %= 1
        rgb = colorsys.hsv_to_rgb(hue, 0.95, 0.95)
        yield tuple(int(x * 255) for x in rgb)


color_generator = generate_distinct_colors()
actor_colors = {}


def get_ansi_color(rgb):
    return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def append_to_log(filename: str, *segments):
    """Write segments to the log file using a stable encoding."""
    try:
        with open(
            filename,
            "a",
            encoding=LOG_FILE_ENCODING,
            errors=LOG_FILE_ERROR_MODE,
            newline="",
        ) as f:
            for segment in segments:
                f.write("" if segment is None else str(segment))
    except OSError as exc:
        print(f"Warning: Unable to write to log '{filename}': {exc}", file=sys.stderr)


def process_and_log_response(reply, actor, filename, contexts, current_model_index):
    global actor_colors

    text = reply.text
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    # Get or generate a color for this actor
    if actor not in actor_colors:
        actor_colors[actor] = get_ansi_color(next(color_generator))

    color = actor_colors[actor]
    bold = "\033[1m"
    reset = "\033[0m"

    # Create a visually distinct header for each actor
    console_header = f"\n{bold}{color}{actor}:{reset}"
    file_header = f"\n### {actor} ###\n"

    print(console_header)
    print(text)

    append_to_log(filename, file_header, text, "\n")
    if reply.reasoning_summary:
        append_to_log(
            filename,
            "\n--- Reasoning Summary (logs only) ---\n",
            reply.reasoning_summary,
            "\n",
        )
    if "^C^C" in text:
        end_message = f"\n{actor} has ended the conversation with ^C^C."
        print(end_message)
        append_to_log(filename, end_message, "\n")
        exit()

    # Add the response to all contexts
    for i, context in enumerate(contexts):
        role = "assistant" if i == current_model_index else "user"
        context.append({"role": role, "content": text})


def cli_conversation(context):
    # Extract the last user message
    last_message = context[-1]["content"]
    # Prepare the payload
    payload = {"messages": [{"role": "user", "content": last_message}]}
    headers = {
        "Authorization": f"Bearer {os.getenv('WORLD_INTERFACE_KEY')}",
        "Content-Type": "application/json",
    }
    # Send POST request to the world-interface
    response = requests.post(
        "http://localhost:3000/v1/chat/completions",
        json=payload,
        headers=headers,
    )
    response.raise_for_status()
    response_data = response.json()
    cli_response = response_data["choices"][0]["message"]["content"]
    return ModelReply(text=cli_response)


if __name__ == "__main__":
    main()
