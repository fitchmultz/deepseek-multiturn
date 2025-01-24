import json
import os
from typing import Dict, List

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
        self.max_auto_iterations = 3  # Default number of auto-responses

    def _stream_request(self, payload: Dict):
        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8").lstrip("data: "))
                    if not chunk.get("choices"):
                        continue

                    delta = chunk["choices"][0].get("delta", {})
                    if delta:  # Only yield if there's actual content
                        if "reasoning_content" in delta and delta["reasoning_content"]:
                            yield ("reasoning", delta["reasoning_content"])
                        elif "content" in delta and delta["content"]:
                            yield ("content", delta["content"])
                except json.JSONDecodeError:
                    continue

        except requests.exceptions.RequestException as e:
            raise Exception(f"API Error: {str(e)}")

    def _generate_auto_response(self) -> str:  # Removed unused parameter
        """
        Generate an auto-response that considers the entire conversation context.
        """
        conversation_history = self.messages.copy()

        # Add instruction as the LAST user message to maintain role alternation
        instruction = {
            "role": "user",
            "content": (
                "Generate the next user message based on the conversation history. "
                "Keep it natural and conversational. Respond ONLY with the user's message text "
                "without any additional commentary or formatting."
            ),
        }
        conversation_history.append(instruction)

        payload = {
            "model": "deepseek-reasoner",
            "messages": conversation_history,
            "stream": True,
        }

        print("\n💭 Generating auto-response with full context:", flush=True)
        print("─" * 80)

        full_response = []
        try:
            has_shown_content_header = False
            for type_, chunk in self._stream_request(payload):
                if type_ == "reasoning":
                    print(chunk, end="", flush=True)
                elif type_ == "content":
                    if not has_shown_content_header:
                        print("\n" + "─" * 80)
                        print("👤 Generated response:", flush=True)
                        has_shown_content_header = True
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)

            print("\n")
            return "".join(full_response)
        except Exception as e:
            print(f"\n❌ Error generating auto-response: {str(e)}")
            return ""

    def chat(self, user_input: str):
        self.messages.append({"role": "user", "content": user_input})

        payload = {
            "model": "deepseek-reasoner",
            "messages": self.messages,
            "stream": True,
        }

        print("\n💭 Reasoning:", flush=True)
        print("─" * 80)
        full_response = []
        full_reasoning = []

        try:
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
            if full_response:
                self.messages.append({"role": "assistant", "content": response_text})
            print("\n")

            # Auto-mode handling
            if self.auto_mode and self.auto_iterations < self.max_auto_iterations:
                self.auto_iterations += 1
                auto_response = self._generate_auto_response()  # Remove parameter
                if auto_response:
                    print(f"\n👤 Auto-user: {auto_response}")
                    self.chat(auto_response)

        except Exception as e:
            self.messages.pop()
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    chat_session = DeepSeekChat()
    print("🤖 DeepSeek Chat - Type 'exit' to quit")
    print("Commands: 'auto' to toggle auto-mode, 'exit' to quit\n")

    while True:
        try:
            user_input = input("👤 You: ")
            if user_input.lower() == "exit":
                break
            elif user_input.lower() == "auto":
                chat_session.auto_mode = not chat_session.auto_mode
                chat_session.auto_iterations = 0
                print(
                    f"\n🔄 Auto-mode {'enabled' if chat_session.auto_mode else 'disabled'}"
                )
                continue
            chat_session.chat(user_input)
        except KeyboardInterrupt:
            break
