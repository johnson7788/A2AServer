"""
Deepseek provider implementation.
"""

import os
import json
import logging
from typing import Dict, List, Any, AsyncGenerator, Optional, Union
from openai import AsyncOpenAI, APIError, RateLimitError
from tenacity import retry, wait_random_exponential, stop_after_attempt, AsyncRetrying


logger = logging.getLogger(__name__)

@retry(wait=wait_random_exponential(min=1, max=3), stop=stop_after_attempt(3), reraise=True)
async def _create_chat_completion_with_retry(client: AsyncOpenAI, **kwargs):
    """
    Helper function to call client.chat.completions.create with retry logic.
    """
    try:
        response = await client.chat.completions.create(**kwargs)
        return response
    except (APIError, RateLimitError) as e:
        logger.warning(f"API call failed, retrying... Error: {e}")
        raise # Re-raise to trigger tenacity's retry mechanism
    except Exception as e:
        logger.error(f"Unexpected error during API call: {e}")
        raise # Re-raise for tenacity to handle or propagate

async def generate_with_deepseek_stream(client: AsyncOpenAI, model_name: str, conversation: List[Dict],
                                    formatted_functions: List[Dict], temperature: Optional[float] = None,
                                    top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> AsyncGenerator:
    """Internal function for streaming generation"""
    try:
        if formatted_functions:
            tools = [{"type": "function", "function": f} for f in formatted_functions]
        else:
            tools = None
        # 使用封装的重试函数
        response = await _create_chat_completion_with_retry(
            client,
            model=model_name,
            messages=conversation,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=tools,
            tool_choice="auto",
            stream=True
        )

        current_tool_calls = []
        current_content = ""

        async for chunk in response:
            logger.debug(f"Deepseek Output chunk: {chunk}")
            delta = chunk.choices[0].delta
            if delta.model_extra:
                if "reasoning_content" in delta.model_extra:
                    yield {"assistant_text": delta.model_extra["reasoning_content"], "tool_calls": [], "is_chunk": True, "token": True, "is_reasoning": True}
            if delta.content:
                # Immediately yield each token without buffering
                yield {"assistant_text": delta.content, "tool_calls": [], "is_chunk": True, "token": True}
                current_content += delta.content

            # Handle tool call updates
            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    # Initialize or update tool call
                    while tool_call.index >= len(current_tool_calls):
                        current_tool_calls.append({
                            "id": "",
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        })
                    
                    current_tool = current_tool_calls[tool_call.index]
                    
                    # Update tool call properties
                    if tool_call.id:
                        current_tool["id"] = tool_call.id
                    
                    if tool_call.function.name:
                        current_tool["function"]["name"] = (
                            current_tool["function"]["name"] + tool_call.function.name
                        )
                    
                    if tool_call.function.arguments:
                        # Properly accumulate JSON arguments
                        current_args = current_tool["function"]["arguments"]
                        new_args = tool_call.function.arguments
                        
                        # Handle special cases for JSON accumulation
                        if new_args.startswith("{") and not current_args:
                            current_tool["function"]["arguments"] = new_args
                        elif new_args.endswith("}") and current_args:
                            # If we're receiving the end of the JSON object
                            if not current_args.endswith("}"):
                                current_tool["function"]["arguments"] = current_args + new_args
                        else:
                            # Middle part of JSON - append carefully
                            current_tool["function"]["arguments"] += new_args

            # If this is the last chunk, yield final state with complete tool calls
            if chunk.choices[0].finish_reason is not None:
                # Clean up and validate tool calls
                final_tool_calls = []
                for tc in current_tool_calls:
                    if tc["id"] and tc["function"]["name"]:
                        try:
                            # Ensure arguments is valid JSON
                            args = tc["function"]["arguments"].strip()
                            if not args or args.isspace():
                                args = "{}"
                            # Parse and validate JSON
                            parsed_args = json.loads(args)
                            tc["function"]["arguments"] = json.dumps(parsed_args)
                            final_tool_calls.append(tc)
                        except json.JSONDecodeError:
                            # If arguments are malformed, try to fix common issues
                            args = tc["function"]["arguments"].strip()
                            # Remove any trailing commas
                            args = args.rstrip(",")
                            # Ensure proper JSON structure
                            if not args.startswith("{"):
                                args = "{" + args
                            if not args.endswith("}"):
                                args = args + "}"
                            try:
                                # Try parsing again after fixes
                                parsed_args = json.loads(args)
                                tc["function"]["arguments"] = json.dumps(parsed_args)
                                final_tool_calls.append(tc)
                            except json.JSONDecodeError:
                                # If still invalid, default to empty object
                                tc["function"]["arguments"] = "{}"
                                final_tool_calls.append(tc)

                yield {
                    "assistant_text": current_content,
                    "tool_calls": final_tool_calls,
                    "is_chunk": False
                }

    except Exception as e:
        yield {"assistant_text": f"DeepSeek error: {str(e)}", "tool_calls": [], "is_chunk": False}

async def generate_with_deepseek_sync(client: AsyncOpenAI, model_name: str, conversation: List[Dict],
                                  formatted_functions: List[Dict], temperature: Optional[float] = None,
                                  top_p: Optional[float] = None, max_tokens: Optional[int] = None) -> Dict:
    """Internal function for non-streaming generation"""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=conversation,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            tools=[{"type": "function", "function": f} for f in formatted_functions],
            tool_choice="auto",
            stream=False
        )

        choice = response.choices[0]
        assistant_text = choice.message.content or ""
        tool_calls = []
        
        if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                if tc.type == 'function':
                    tool_call = {
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}"
                        }
                    }
                    # Ensure arguments is valid JSON
                    try:
                        json.loads(tool_call["function"]["arguments"])
                    except json.JSONDecodeError:
                        tool_call["function"]["arguments"] = "{}"
                    tool_calls.append(tool_call)
        return {"assistant_text": assistant_text, "tool_calls": tool_calls}

    except APIError as e:
        return {"assistant_text": f"DeepSeek API error: {str(e)}", "tool_calls": []}
    except RateLimitError as e:
        return {"assistant_text": f"DeepSeek rate limit: {str(e)}", "tool_calls": []}
    except Exception as e:
        return {"assistant_text": f"Unexpected DeepSeek error: {str(e)}", "tool_calls": []}

async def generate_with_deepseek(conversation: List[Dict], model_cfg: Dict,
                             all_functions: List[Dict], stream: bool = False) -> Union[Dict, AsyncGenerator]:
    """
    Generate text using OpenAI's API.
    Args:
        conversation: The conversation history
        model_cfg: Configuration for the model
        all_functions: Available functions for the model to call
        stream: Whether to stream the response
    Returns:
        If stream=False: Dict containing assistant_text and tool_calls
        If stream=True: AsyncGenerator yielding chunks of assistant text and tool calls
    """
    api_key = model_cfg.get("apiKey") or os.getenv("DEEPSEEK_API_KEY")
    if "apiBase" in model_cfg:
        client = AsyncOpenAI(api_key=api_key, base_url=model_cfg["apiBase"])
    else:
        client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    model_name = model_cfg["model"]
    temperature = model_cfg.get("temperature", None)
    top_p = model_cfg.get("top_p", None)
    max_tokens = model_cfg.get("max_tokens", None)

    # Format functions for OpenAI API
    formatted_functions = []
    for func in all_functions:
        formatted_func = {
            "name": func["name"],
            "description": func["description"],
            "parameters": func["parameters"]
        }
        formatted_functions.append(formatted_func)

    if stream:
        return generate_with_deepseek_stream(
            client, model_name, conversation, formatted_functions,
            temperature, top_p, max_tokens
        )
    else:
        return await generate_with_deepseek_sync(
            client, model_name, conversation, formatted_functions,
            temperature, top_p, max_tokens
        )
