import json
import os
import pickle
import re
from datetime import datetime
from pathlib import Path
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
        self.max_auto_retries = 2
        self.show_reasoning = False
        self.save_dir = Path("chat_history")
        self.save_dir.mkdir(exist_ok=True)

        self._auto_instruction = (
            "Generate the next USER message based on the conversation history. "
            "Keep it natural and match the user's message style. Be as LENGTHY as "
            "you desire. Being CONCISE is fine too! Respond ONLY with "
            "the user's message text WITHOUT any additional commentary. You can "
            "add markdown formatting if you want, but it's not required."
        )

    def save_conversation(self, filename=None) -> None:
        """Save current conversation state to disk"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chat_session_{timestamp}.pkl"

        save_path = self.save_dir / filename

        # Create state dictionary
        state = {
            "messages": self.messages,
            "auto_mode": self.auto_mode,
            "auto_iterations": self.auto_iterations,
            "show_reasoning": self.show_reasoning,
        }

        with open(save_path, "wb") as f:
            pickle.dump(state, f)

        console.print(
            Panel(f"💾 Conversation saved to {save_path}", border_style="green")
        )

    def load_conversation(self, filename: str) -> None:
        """Load conversation state from disk"""
        load_path = self.save_dir / filename

        if not load_path.exists():
            console.print(Panel(f"❌ File not found: {load_path}", border_style="red"))
            return

        try:
            with open(load_path, "rb") as f:
                state = pickle.load(f)

            self.messages = state["messages"]
            self.auto_mode = state["auto_mode"]
            self.auto_iterations = state["auto_iterations"]
            self.show_reasoning = state["show_reasoning"]

            console.print(
                Panel("📂 Conversation loaded successfully", border_style="green")
            )

            # Display last few messages for context
            last_msgs = self.messages[-3:] if len(self.messages) > 3 else self.messages
            console.print(Panel("Last few messages:", border_style="cyan"))
            for msg in last_msgs:
                style = self.PANEL_STYLES[
                    "user" if msg["role"] == "user" else "assistant"
                ]
                console.print(Panel(Markdown(msg["content"]), **style))

        except Exception as e:
            console.print(
                Panel(f"❌ Error loading conversation: {str(e)}", border_style="red")
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

                data_content = raw_line[5:].strip()
                if data_content == "[DONE]":
                    break

                if not data_content:
                    continue

                try:
                    buffer += data_content
                    if not (buffer.startswith("{") and buffer.endswith("}")):
                        continue

                    chunk = json.loads(buffer)
                    buffer = ""

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
                    buffer = ""
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
                            has_shown_content = True

                        content_buffer.append(chunk)
                        full_response.append(chunk)

                        panel = Panel(
                            Markdown("".join(content_buffer)),
                            **{**self.PANEL_STYLES["user"], "title": "👤 Auto-User"},
                        )
                        live.update(panel)

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
            self._validate_message_sequence("user")
            self.messages.append(
                {"role": "user", "content": user_input, "is_auto_generated": False}
            )

            console.print(Panel(Markdown(user_input), **self.PANEL_STYLES["user"]))

            self.auto_iterations = 0

            response_text = self._process_chat_round()
            if not response_text:
                return

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

                    response_text = self._process_chat_round()
                    if not response_text:
                        console.print(
                            Panel(
                                "⚠️ Assistant response failed, stopping auto-mode",
                                border_style="yellow",
                            )
                        )
                        break

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

            # Create a static panel that we'll update
            panel = None

            # Calculate available height for the live display
            term_height = console.height
            max_display_height = (
                term_height - 4
            )  # Reserve space for borders and padding

            with Live(
                Panel("", title="🤖 Assistant", border_style="blue", style="blue"),
                console=console,
                refresh_per_second=4,
                vertical_overflow="crop",  # Change to crop
                auto_refresh=False,  # Disable auto-refresh
                transient=True,  # Use transient mode
            ) as live:
                for type_, chunk in self._stream_request(payload):
                    if type_ == "reasoning":
                        if self.show_reasoning:
                            console.print(chunk, style="yellow", end="")
                    elif type_ == "content":
                        content_buffer.append(chunk)
                        full_response.append(chunk)

                        # Create complete content string
                        current_content = "".join(content_buffer)

                        # Only create new panel if content actually changed
                        if panel is None or panel.renderable.markup != current_content:
                            panel = Panel(
                                Markdown(current_content),
                                title="🤖 Assistant",
                                border_style="blue",
                                style="blue",
                                height=max_display_height,
                            )
                            live.update(panel)
                            live.refresh()  # Explicit refresh

                # Final render - print the complete response without Live
                final_content = "".join(content_buffer)
                console.print(
                    Panel(
                        Markdown(final_content),
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
        "'auto' to toggle auto-mode, 'reason' to toggle reasoning, "
        "'save' to save conversation, 'load <filename>' to load conversation, 'exit' to quit",
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
            elif user_input.lower().strip() == "save":
                chat_session.save_conversation()
                continue
            elif user_input.lower().startswith("load "):
                filename = user_input[5:].strip()
                chat_session.load_conversation(filename)
                continue

            chat_session.chat(user_input)

    except KeyboardInterrupt:
        console.print(Panel("👋 Session ended", border_style="cyan"))
