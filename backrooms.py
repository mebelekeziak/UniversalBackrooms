import anthropic
from openai import OpenAI
import json
import datetime
import os
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
    "gpt-4o": {
        "api_name": "gpt-4o",
        "display_name": "GPT-4o",
        "company": "openai",
        "is_reasoning": False,
    },
    "gpt-5": {
        "api_name": "gpt-5",
        "display_name": "GPT-5",
        "company": "openai",
        "is_reasoning": True,
    },
    "gpt-5-mini": {
        "api_name": "gpt-5-mini",
        "display_name": "GP5-5 Mini",
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

ANSI_RESET = "\033[0m"
ANSI_BLUE = "\033[94m"


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


def load_template(template_name, models, actor_labels=None):
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
                if actor_labels and i < len(actor_labels):
                    actors.append(actor_labels[i])
                else:
                    actors.append("CLI")
            else:
                companies.append(MODEL_INFO[model]["company"])
                if actor_labels and i < len(actor_labels):
                    actors.append(actor_labels[i])
                else:
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


def _enable_windows_ansi():
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        # If enabling ANSI escapes fails we silently continue; menus still work albeit without color.
        pass


def _supports_color():
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _colorize(text: str, color: str) -> str:
    if not _supports_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def _clear_lines(count: int):
    if count <= 0 or not sys.stdout.isatty():
        return
    for _ in range(count):
        sys.stdout.write("\033[F\033[K")
    sys.stdout.flush()


def _read_nav_key():
    if os.name == "nt":
        import msvcrt

        while True:
            ch = msvcrt.getwch()
            if ch == "\x03":
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):
                return "ENTER"
            if ch in (" ",):
                return "ENTER"
            lower = ch.lower()
            if lower == "w":
                return "UP"
            if lower == "s":
                return "DOWN"
            if lower == "a":
                return "LEFT"
            if lower == "d":
                return "RIGHT"
            if ch in ("\x00", "\xe0"):
                ch2 = msvcrt.getwch()
                mapping = {"H": "UP", "P": "DOWN", "K": "LEFT", "M": "RIGHT"}
                mapped = mapping.get(ch2.upper())
                if mapped:
                    return mapped
            # Ignore other keys and continue the loop
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                ch = sys.stdin.read(1)
                if ch == "\x03":
                    raise KeyboardInterrupt
                if ch in ("\r", "\n", " "):
                    return "ENTER"
                lower = ch.lower()
                if lower == "w":
                    return "UP"
                if lower == "s":
                    return "DOWN"
                if lower == "a":
                    return "LEFT"
                if lower == "d":
                    return "RIGHT"
                if ch == "\x1b":
                    second = sys.stdin.read(1)
                    if second in ("[", "O"):
                        third = sys.stdin.read(1)
                        mapping = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}
                        mapped = mapping.get(third.upper())
                        if mapped:
                            return mapped
                # Ignore anything else
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _render_menu(prompt: str, options, current_index: int, default_value):
    lines = [
        "",
        prompt,
        "Use ↑/↓ arrows or W/S to move; press Enter to select.",
    ]
    for idx, (value, label) in enumerate(options):
        marker = ">" if idx == current_index else " "
        suffix = " (default)" if default_value is not None and value == default_value else ""
        text = f"{label}{suffix}"
        if idx == current_index:
            text = _colorize(text, ANSI_BLUE)
        lines.append(f" {marker} {text}")

    output = "\n".join(lines)
    print(output, flush=True)
    return len(lines)


def _prompt_menu(prompt: str, options, default_value=None):
    """Prompt the user to select from an interactive list using arrow or WASD keys."""
    if not options:
        raise ValueError("No options provided for selection.")

    option_values = [value for value, _ in options]
    current_index = 0
    if default_value is not None and default_value in option_values:
        current_index = option_values.index(default_value)

    while True:
        lines_rendered = _render_menu(prompt, options, current_index, default_value)
        key = _read_nav_key()

        if key in {"UP", "LEFT"}:
            _clear_lines(lines_rendered)
            current_index = (current_index - 1) % len(options)
            continue
        if key in {"DOWN", "RIGHT"}:
            _clear_lines(lines_rendered)
            current_index = (current_index + 1) % len(options)
            continue
        if key == "ENTER":
            print()
            return option_values[current_index]

        # Unhandled keys simply re-render the menu
        _clear_lines(lines_rendered)


def _prompt_int(prompt: str, default_value=None, minimum=None):
    """Prompt the user for an integer with optional default and minimum."""
    while True:
        suffix = f" [{default_value}]" if default_value is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            if default_value is not None:
                return default_value
            print("Please enter a value.")
            continue
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum}.")
            continue

        return value


def _prompt_yes_no(prompt: str, default: bool = False) -> bool:
    """Prompt for yes/no using the interactive menu."""
    options = [("yes", "Yes"), ("no", "No")]
    default_value = "yes" if default else "no"
    selection = _prompt_menu(prompt, options, default_value=default_value)
    return selection == "yes"


def run_interactive_builder():
    if not sys.stdin.isatty():
        print("Interactive setup requires a TTY. Please run this script from a terminal.")
        sys.exit(1)

    _enable_windows_ansi()

    print("=== Universal Backrooms Setup ===")
    model_options = []
    for key, info in MODEL_INFO.items():
        label = f"{info['display_name']} ({key}) - {info['company'].capitalize()}"
        model_options.append((key, label))
    model_options.append(("cli", "CLI Interface (cli)"))

    selections = []
    for idx, suffix in enumerate(["A", "B"], start=1):
        default_model = "opus"
        if default_model not in [value for value, _ in model_options]:
            default_model = model_options[0][0]
        model_key = _prompt_menu(
            f"Select model {suffix}", model_options, default_value=default_model
        )

        if model_key == "cli":
            actor_name = f"CLI {suffix}"
        else:
            display_name = MODEL_INFO[model_key]["display_name"]
            actor_name = f"{display_name} {suffix}"

        selections.append({"key": model_key, "actor": actor_name})

    available_templates = get_available_templates()
    if not available_templates:
        print("No templates found in ./templates. Please add a template before running.")
        sys.exit(1)
    template_choices = [(name, name) for name in available_templates]
    template = _prompt_menu(
        "Choose a conversation template",
        template_choices,
        default_value=available_templates[0] if available_templates else None,
    )

    max_turns_input = _prompt_int(
        "Maximum number of turns (0 for unlimited)", default_value=10, minimum=0
    )
    max_turns = None if max_turns_input == 0 else max_turns_input

    selected_model_keys = [item["key"] for item in selections]
    selected_api_models = [
        MODEL_INFO[key]["api_name"]
        for key in selected_model_keys
        if key in MODEL_INFO
    ]

    any_openai = any(
        key in MODEL_INFO and MODEL_INFO[key]["company"] == "openai"
        for key in selected_model_keys
    )
    has_reasoning_models = any(api in REASONING_MODELS for api in selected_api_models)

    include_encrypted_reasoning = False
    reasoning_effort = "medium"
    openai_max_output_tokens = 14048

    if any_openai:
        openai_max_output_tokens = _prompt_int(
            "Max OpenAI output tokens", default_value=14048, minimum=1
        )

    if has_reasoning_models:
        include_encrypted_reasoning = _prompt_yes_no(
            "Request encrypted reasoning traces for GPT-5 family models?",
            default=False,
        )
        reasoning_effort = _prompt_menu(
            "Reasoning effort for GPT-5 family models",
            [("low", "Low"), ("medium", "Medium"), ("high", "High")],
            default_value="medium",
        )
        print(
            "\nDisclaimer: OpenAI only returns reasoning summaries for verified organisations."
            " If your organisation is unverified you will not receive reasoning summaries, but encrypted reasoning is still available."
        )

    print("\nSetup complete. Starting conversation...\n")

    return {
        "models": selected_model_keys,
        "actor_labels": [item["actor"] for item in selections],
        "template": template,
        "max_turns": max_turns,
        "include_encrypted_reasoning": include_encrypted_reasoning,
        "reasoning_effort": reasoning_effort,
        "openai_max_output_tokens": openai_max_output_tokens,
    }


def main():
    global anthropic_client
    global openai_client
    try:
        builder_config = run_interactive_builder()
    except (KeyboardInterrupt, EOFError):
        print("\nSetup cancelled.")
        return

    models = builder_config["models"]
    actor_labels = builder_config["actor_labels"]
    template_name = builder_config["template"]
    max_turns = builder_config["max_turns"]
    include_encrypted_reasoning = builder_config["include_encrypted_reasoning"]
    reasoning_effort = builder_config["reasoning_effort"]
    openai_max_output_tokens = builder_config["openai_max_output_tokens"]

    lm_models = []
    lm_display_names = []
    companies = []

    for index, model_key in enumerate(models):
        actor_label = actor_labels[index] if index < len(actor_labels) else ""
        if model_key.lower() == "cli":
            lm_models.append("cli")
            companies.append("CLI")
            lm_display_names.append(actor_label or "CLI")
        else:
            info = MODEL_INFO.get(model_key)
            if not info:
                print(f"Error: Model '{model_key}' not found in MODEL_INFO.")
                sys.exit(1)
            lm_models.append(info["api_name"])
            companies.append(info["company"])
            lm_display_names.append(actor_label or f"{info['display_name']} {index + 1}")

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

    configs = load_template(template_name, models, actor_labels)

    openai_settings = {
        "include_encrypted_reasoning": include_encrypted_reasoning,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": openai_max_output_tokens,
    }

    assert len(models) == len(
        configs
    ), f"Number of LMs ({len(models)}) does not match the number of elements in the template ({len(configs)})"

    system_prompts = [config.get("system_prompt", "") for config in configs]
    contexts = [config.get("context", []) for config in configs]
    logs_folder = "BackroomsLogs"
    os.makedirs(logs_folder, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{logs_folder}/{'_'.join(models)}_{template_name}_{timestamp}.txt"

    turn = 0
    while max_turns is None or turn < max_turns:
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

    if max_turns is not None:
        print(f"\nReached maximum number of turns ({max_turns}). Conversation ended.")
        append_to_log(
            filename,
            f"\nReached maximum number of turns ({max_turns}). Conversation ended.\n",
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
