from flask import Flask, request, jsonify
from qachain import QAChainTool

app = Flask(__name__)
qa_tool = QAChainTool()

@app.route('/tool/retrieve_chunks', methods=['POST'])
def retrieve_chunks():
    data = request.get_json()
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Query cannot be empty."}), 400

    response = qa_tool.query_relevant_chunks(query)

    if "error" in response:
        return jsonify({"error": response["error"]}), 500

    return jsonify(response)

@app.route('/greeting/<name>', methods=['GET'])
def get_greeting(name):
    return jsonify({"greeting": f"Hello, {name}!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
