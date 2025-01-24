import json
import os
import re
from typing import Dict, Generator, List, Optional

import requests


class DeepSeekChat:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("Please set DEEPSEEK_API_KEY environment variable")

        self.base_url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        self.messages: List[Dict] = []
        self.auto_mode = False
        self.auto_iterations = 0
        self.max_auto_iterations = 3
        self._auto_instruction = (
            "Generate the next user message based on the conversation history. "
            "Keep it natural and conversational. Respond ONLY with the user's message text "
            "without any additional commentary or formatting."
        )

    def _validate_message_sequence(self, new_role: str) -> None:
        """Ensure proper alternation between user and assistant roles."""
        if self.messages:
            last_role = self.messages[-1]["role"]
            if last_role == new_role:
                raise ValueError(f"Consecutive {new_role} messages detected")

    def _clean_auto_messages(self) -> None:
        """Remove any auto-generated messages from history"""
        self.messages = [
            msg for msg in self.messages if not msg.get("is_auto_generated")
        ]

    def _stream_request(self, payload: Dict) -> Generator[tuple[str, str], None, None]:
        """Improved SSE parsing with complete JSON validation"""
        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, stream=True
            )
            response.raise_for_status()

            buffer = ""
            for line in response.iter_lines():
                if not line:
                    continue

                raw_line = line.decode("utf-8").strip()
                if not raw_line.startswith("data:"):
                    continue

                # Extract content and handle server keep-alives
                data_content = raw_line[5:].strip()  # Remove 'data:' prefix
                if data_content == "[DONE]":
                    break

                if not data_content:
                    continue

                try:
                    # Handle potential partial JSON
                    buffer += data_content
                    if not (buffer.startswith("{") and buffer.endswith("}")):
                        continue  # Wait for complete JSON object

                    chunk = json.loads(buffer)
                    buffer = ""  # Reset buffer after successful parse

                    if not chunk.get("choices"):
                        continue

                    delta = chunk["choices"][0].get("delta", {})
                    if delta:
                        if delta.get("reasoning_content"):
                            yield ("reasoning", delta["reasoning_content"])
                        elif delta.get("content"):
                            yield ("content", delta["content"])

                except json.JSONDecodeError as e:
                    print(f"JSON parse error: {str(e)} in data: {data_content}")
                    buffer = ""  # Reset buffer on error
                    continue

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API Error: {str(e)}")

    def _generate_auto_response(self, verbose: bool = True) -> Optional[str]:
        """Generate auto-response with optional output suppression"""
        try:
            self._validate_message_sequence("user")
            conversation_history = [
                msg
                for msg in self.messages.copy()
                if msg["content"] != self._auto_instruction
            ]

            conversation_history.append(
                {
                    "role": "user",
                    "content": self._auto_instruction,
                    "is_auto_generated": True,
                }
            )

            payload = {
                "model": "deepseek-reasoner",
                "messages": conversation_history,
                "stream": True,
            }

            if verbose:
                print("\n💭 Generating auto-response with full context:", flush=True)
                print("─" * 80)

            full_response = []
            has_shown_content_header = False

            for type_, chunk in self._stream_request(payload):
                if type_ == "reasoning":
                    if verbose:
                        print(chunk, end="", flush=True)
                elif type_ == "content":
                    full_response.append(chunk)
                    if verbose and not has_shown_content_header:
                        print("\n" + "─" * 80)
                        print("👤 Generated response:", flush=True)
                        has_shown_content_header = True
                    if verbose:
                        print(chunk, end="", flush=True)

            if verbose:
                print("\n")
            return "".join(full_response).strip()

        except Exception as e:
            if verbose:
                print(f"\n❌ Error generating auto-response: {str(e)}")
            return None

    def toggle_auto_mode(self) -> None:
        """Safely toggle auto-mode with state cleanup"""
        self.auto_mode = not self.auto_mode
        self.auto_iterations = 0
        if self.auto_mode:
            self._clean_auto_messages()
        print(f"\n🔄 Auto-mode {'enabled' if self.auto_mode else 'disabled'}")

    def chat(self, user_input: str) -> None:
        """Main chat method with iterative auto-response handling"""
        try:
            # Add user message with validation
            self._validate_message_sequence("user")
            self.messages.append(
                {"role": "user", "content": user_input, "is_auto_generated": False}
            )

            # Generate initial response
            response_text = self._process_chat_round()
            if not response_text:
                return

            # Handle auto-responses iteratively with loop control
            while (
                self.auto_mode
                and self.auto_iterations < self.max_auto_iterations
                and response_text is not None
            ):
                self.auto_iterations += 1  # Increment before processing
                print(
                    f"\n🔁 Auto-iteration: {self.auto_iterations}/{self.max_auto_iterations}"
                )

                auto_response = self._generate_auto_response(verbose=False)
                if not auto_response:
                    break

                print(f"\n👤 Auto-user: {auto_response}")
                self.messages.append(
                    {
                        "role": "user",
                        "content": auto_response,
                        "is_auto_generated": True,
                    }
                )

                # Process assistant response
                response_text = self._process_chat_round()
                if not response_text:
                    break

        except Exception as e:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            print(f"\n❌ Error: {str(e)}")

    def _process_chat_round(self) -> Optional[str]:
        """Handle a single round of chat interaction"""
        try:
            self._validate_message_sequence("assistant")
            payload = {
                "model": "deepseek-reasoner",
                "messages": self.messages,
                "stream": True,
            }

            print("\n💭 Reasoning:", flush=True)
            print("─" * 80)
            full_response = []
            full_reasoning = []
            has_shown_content_header = False

            for type_, chunk in self._stream_request(payload):
                if type_ == "reasoning":
                    print(chunk, end="", flush=True)
                    full_reasoning.append(chunk)
                elif type_ == "content":
                    if not has_shown_content_header:
                        print("\n" + "─" * 80)
                        print("🤖 Response:", flush=True)
                        has_shown_content_header = True
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)

            response_text = "".join(full_response)
            if response_text:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "is_auto_generated": False,
                    }
                )
            print("\n")
            return response_text

        except Exception as e:
            print(f"\n❌ Processing error: {str(e)}")
            return None


if __name__ == "__main__":
    chat_session = DeepSeekChat()
    print("🤖 DeepSeek Chat - Type 'exit' to quit")
    print("Commands: 'auto' to toggle auto-mode, 'exit' to quit\n")

    try:
        while True:
            user_input = input("👤 You: ").strip()
            if not user_input:
                continue

            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "auto":
                chat_session.toggle_auto_mode()
                continue

            chat_session.chat(user_input)

    except KeyboardInterrupt:
        print("\n\n👋 Session ended")
