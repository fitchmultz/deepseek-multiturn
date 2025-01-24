# app.py
import json

from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from deepseek import DeepSeekChat  # Import your existing class

app = Flask(__name__)
chat_session = DeepSeekChat()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message")

    def generate():
        try:
            # Process initial user message
            chat_session._validate_message_sequence("user")
            chat_session.messages.append(
                {"role": "user", "content": user_input, "is_auto_generated": False}
            )

            auto_iterations = 0
            current_message = user_input

            while True:
                # Process assistant response
                payload = {
                    "model": "deepseek-reasoner",
                    "messages": chat_session.messages,
                    "stream": True,
                }

                # Stream assistant response
                response_chunks = []
                for type_, chunk in chat_session._stream_request(payload):
                    if type_ == "content":
                        response_chunks.append(chunk)
                        yield f"data: {json.dumps({'type': 'content', 'chunk': chunk})}\n\n"
                    elif type_ == "reasoning" and chat_session.show_reasoning:
                        yield f"data: {json.dumps({'type': 'reasoning', 'chunk': chunk})}\n\n"

                # Save complete assistant response
                complete_response = "".join(response_chunks)
                chat_session.messages.append(
                    {
                        "role": "assistant",
                        "content": complete_response,
                        "is_auto_generated": False,
                    }
                )

                # Check if we should continue with auto-mode
                if (
                    not chat_session.auto_mode
                    or auto_iterations >= chat_session.max_auto_iterations
                ):
                    break

                # Generate auto-user response
                auto_response = chat_session._generate_auto_response()
                if not auto_response:
                    break

                # Yield auto-user message
                yield f"data: {json.dumps({'type': 'auto_user', 'content': auto_response})}\n\n"

                # Add auto-user message to history
                chat_session.messages.append(
                    {
                        "role": "user",
                        "content": auto_response,
                        "is_auto_generated": True,
                    }
                )

                auto_iterations += 1

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/toggle_reasoning", methods=["POST"])
def toggle_reasoning():
    chat_session.toggle_reasoning()
    return jsonify({"show_reasoning": chat_session.show_reasoning})


@app.route("/toggle_auto", methods=["POST"])
def toggle_auto():
    chat_session.toggle_auto_mode()
    return jsonify({"auto_mode": chat_session.auto_mode})


if __name__ == "__main__":
    app.run(debug=True)
