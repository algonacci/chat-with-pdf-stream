import os
from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from embedchain import App
from embedchain.config import BaseLlmConfig
from embedchain.helpers.callbacks import StreamingStdOutCallbackHandlerYield, generate
import queue
import threading


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


app = Flask(__name__)
RAG_app = App.from_config(config_path="config.yaml")

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        file = request.files['pdfFile']
        if file and file.filename.endswith('.pdf'):
                try:
                    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                    file.save(file_path)

                    RAG_app.add("pdf_file", file_path),

                    return render_template("chat.html")
                
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
        else:
                return jsonify({"error": "Only PDF files are allowed"}), 400


@app.route("/rag")
def rag():
    msg = request.args.get('msg')

    q = queue.Queue()

    def app_response(result):
            llm_config = RAG_app.llm.config.as_dict()
            if "http_client" in llm_config:
                del llm_config["http_client"]
                del llm_config["http_async_client"]
            llm_config["callbacks"] = [StreamingStdOutCallbackHandlerYield(q=q)]
            config = BaseLlmConfig(**llm_config)
            answer = RAG_app.chat(msg, config=config)
            result["answer"] = answer

    results = {}
    thread = threading.Thread(target=app_response, args=(results,))
    thread.start()

    def generate_response():
            full_response = ""
            # Streaming chunks from the queue as they become available
            for answer_chunk in generate(q):
                full_response += answer_chunk
                
                yield answer_chunk  # Directly yielding the chunk to the client
            print(full_response)
            yield

    return Response(stream_with_context(generate_response()), content_type='text/event-stream')




if __name__ == "__main__":
    app.run()