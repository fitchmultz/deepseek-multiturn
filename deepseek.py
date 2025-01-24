import json
import os
import re
from typing import Dict, Generator, List, Optional

import requests
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()


class DeepSeekChat:
    PANEL_STYLES = {
        "user": {"title": "👤 You", "border_style": "green", "style": "green"},
        "assistant": {"title": "🤖 Assistant", "border_style": "blue", "style": "blue"},
        "reasoning": {
            "title": "💭 Reasoning",
            "border_style": "yellow",
            "style": "yellow",
        },
    }

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
        self.auto_mode = True
        self.auto_iterations = 0
        self.max_auto_iterations = 3
        self.max_auto_retries = 2  # New: Maximum retries for auto-mode errors
        self.show_reasoning = False
        self._auto_instruction = (
            "Generate the next USER message based on the conversation history. "
            "Keep it natural and match the user's message style. Be as LENGTHY as "
            "you desire. Being CONCISE is fine too! Respond ONLY with "
            "the user's message text WITHOUT any additional commentary. You can "
            "add markdown formatting if you want, but it's not required."
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

    def toggle_auto_mode(self) -> None:
        """Safely toggle auto-mode with state cleanup"""
        self.auto_mode = not self.auto_mode
        self.auto_iterations = 0
        if self.auto_mode:
            self._clean_auto_messages()

        status = Text()
        status.append("🔄 Auto-mode ", style="cyan")
        status.append("enabled" if self.auto_mode else "disabled", style="bold cyan")
        status.append(" + ", style="cyan")
        status.append(f"Max auto-iterations: {self.max_auto_iterations}", style="bold")
        console.print(Panel(status, border_style="cyan"))

    def toggle_reasoning(self) -> None:
        """Toggle visibility of assistant reasoning output"""
        self.show_reasoning = not self.show_reasoning
        status = Text()
        status.append("🔄 Assistant reasoning display ", style="cyan")
        status.append(
            "enabled" if self.show_reasoning else "disabled", style="bold cyan"
        )
        console.print(Panel(status, border_style="cyan"))

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
                    console.print(
                        f"[red]JSON parse error: {str(e)} in data: {data_content}[/red]"
                    )
                    buffer = ""  # Reset buffer on error
                    continue

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API Error: {str(e)}")

    def _generate_auto_response(self) -> Optional[str]:
        """Generate auto-response with streaming markdown rendering"""
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

            if self.show_reasoning:
                console.print(
                    Panel(
                        "💭 Generating auto-response with full context",
                        border_style="cyan",
                    )
                )
                console.print("─" * 80)

            full_response = []
            content_buffer = []
            has_shown_content = False

            self.suppress_auto_display = True

            with Live(
                console=console, refresh_per_second=4, vertical_overflow="visible"
            ) as live:
                for type_, chunk in self._stream_request(payload):
                    if type_ == "reasoning":
                        if self.show_reasoning:
                            console.print(
                                chunk,
                                style=self.PANEL_STYLES["reasoning"]["style"],
                                end="",
                            )
                    elif type_ == "content":
                        if not has_shown_content:
                            # Remove explicit header prints
                            has_shown_content = True

                        content_buffer.append(chunk)
                        full_response.append(chunk)

                        # Update with combined panel
                        panel = Panel(
                            Markdown("".join(content_buffer)),
                            **{**self.PANEL_STYLES["user"], "title": "👤 Auto-User"},
                        )
                        live.update(panel)

                # Final update with complete panel
                if has_shown_content:
                    live.update(
                        Panel(
                            Markdown("".join(content_buffer)),
                            **{**self.PANEL_STYLES["user"], "title": "👤 Auto-User"},
                        )
                    )

            self.suppress_auto_display = False
            return "".join(full_response).strip()
        except Exception as e:
            console.print(
                Panel(
                    f"❌ Error generating auto-response: {str(e)}", border_style="red"
                )
            )
            return None

    def chat(self, user_input: str) -> None:
        """Main chat method with iterative auto-response handling"""
        try:
            # Add user message with validation
            self._validate_message_sequence("user")
            self.messages.append(
                {"role": "user", "content": user_input, "is_auto_generated": False}
            )

            # Display user input as markdown
            console.print(Panel(Markdown(user_input), **self.PANEL_STYLES["user"]))

            # Reset auto-iteration counter for new user input
            self.auto_iterations = 0

            # Generate initial response
            response_text = self._process_chat_round()
            if not response_text:
                return

            # Handle auto-responses iteratively with loop control and error handling
            retry_count = 0
            while (
                self.auto_mode
                and self.auto_iterations < self.max_auto_iterations
                and response_text is not None
            ):
                try:
                    auto_response = self._generate_auto_response()
                    if not auto_response:
                        console.print(
                            Panel(
                                "⚠️ Auto-response generation failed, stopping auto-mode",
                                border_style="yellow",
                            )
                        )
                        break

                    self.auto_iterations += 1
                    console.print(
                        Panel(
                            f"🔄 Auto-iteration: {self.auto_iterations}/{self.max_auto_iterations}",
                            border_style="cyan",
                        )
                    )
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
                        console.print(
                            Panel(
                                "⚠️ Assistant response failed, stopping auto-mode",
                                border_style="yellow",
                            )
                        )
                        break

                    # Reset retry count on successful iteration
                    retry_count = 0

                except Exception as e:
                    retry_count += 1
                    if retry_count > self.max_auto_retries:
                        console.print(
                            Panel(
                                f"❌ Max retries ({self.max_auto_retries}) exceeded in auto-mode, stopping",
                                border_style="red",
                            )
                        )
                        break
                    console.print(
                        Panel(
                            f"⚠️ Auto-mode error (attempt {retry_count}/{self.max_auto_retries}): {str(e)}",
                            border_style="yellow",
                        )
                    )
                    continue

        except Exception as e:
            if self.messages and self.messages[-1]["role"] == "user":
                self.messages.pop()
            console.print(Panel(f"❌ Error: {str(e)}", border_style="red"))

    def _process_chat_round(self) -> Optional[str]:
        """Handle chat interaction with streaming markdown rendering"""
        try:
            self._validate_message_sequence("assistant")
            payload = {
                "model": "deepseek-reasoner",
                "messages": self.messages,
                "stream": True,
            }

            if self.show_reasoning:
                console.print(Panel("💭 Reasoning:", border_style="yellow"))
                console.print("─" * 80)

            full_response = []
            content_buffer = []
            has_content = False

            with Live(
                console=console, refresh_per_second=4, vertical_overflow="visible"
            ) as live:
                for type_, chunk in self._stream_request(payload):
                    if type_ == "reasoning":
                        if self.show_reasoning:
                            console.print(chunk, style="yellow", end="")
                    elif type_ == "content":
                        content_buffer.append(chunk)
                        full_response.append(chunk)

                        # Create panel with current content
                        panel = Panel(
                            Markdown("".join(content_buffer)),
                            title="🤖 Assistant",
                            border_style="blue",
                            style="blue",
                        )
                        live.update(panel)

                # Final update with complete panel
                if content_buffer:
                    live.update(
                        Panel(
                            Markdown("".join(content_buffer)),
                            title="🤖 Assistant",
                            border_style="blue",
                            style="blue",
                        )
                    )

            response_text = "".join(full_response)
            if response_text:
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": response_text,
                        "is_auto_generated": False,
                    }
                )

            return response_text

        except Exception as e:
            console.print(Panel(f"❌ Processing error: {str(e)}", border_style="red"))
            return None


if __name__ == "__main__":
    chat_session = DeepSeekChat()
    title = Text()
    title.append("🤖 DeepSeek Chat", style="bold")
    title.append(" - Type 'exit' to quit\n", style="dim")
    title.append("Commands: ", style="bold")
    title.append(
        "'auto' to toggle auto-mode, 'reason' to toggle reasoning, 'exit' to quit",
        style="italic",
    )
    console.print(Panel(title, border_style="blue"))

    try:
        while True:
            user_input = console.input("[green]👤 You: [/green]").strip()
            if not user_input:
                continue

            if user_input.lower().strip() == "exit":
                break
            elif user_input.lower().strip() == "auto":
                chat_session.toggle_auto_mode()
                continue
            elif user_input.lower().strip() in ("reason", "reasoning"):
                chat_session.toggle_reasoning()
                continue

            chat_session.chat(user_input)

    except KeyboardInterrupt:
        console.print(Panel("👋 Session ended", border_style="cyan"))
