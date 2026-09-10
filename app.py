import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.credentials import Credentials

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

def _build_model():
    credentials = Credentials(
        url=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        api_key=os.getenv("IBM_CLOUD_API_KEY")
    )
    
    parameters = {
        GenParams.MAX_NEW_TOKENS: 500,
        GenParams.TEMPERATURE: 0.7,
    }
    
    project_id = os.getenv("WATSONX_PROJECT_ID")
    if not project_id:
        raise ValueError("WATSONX_PROJECT_ID is missing in the .env file")

    # Intha edathula thaan correct-ana supported model name potruken!
    return ModelInference(
        model_id="mistralai/mistral-small-3-1-24b-instruct-2503", 
        project_id=project_id,
        credentials=credentials,
        params=parameters
    )

try:
    _model = _build_model()
    print("Model initialized successfully! AI is ready.")
except Exception as e:
    print(f"Model initialization error: {e}")
    _model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
@app.route('/chat', methods=['POST'])
def chat():
    if not _model:
        return jsonify({"error": "Model not initialized."}), 500
    
    data = request.json or {}
    user_message = data.get('message', '')
    role = data.get('role', 'software engineer')
    
    # AI-ku strict-ana system instructions kudukrom
    full_prompt = f"""You are a professional technical interviewer conducting a mock interview for a {role} position. 
    Your task is to interview the candidate step by step. Ask only one question at a time and wait for their answer. Do not answer for them.
    
    Candidate says: {user_message}
    Interviewer reply:"""
    
    try:
        response = _model.generate_text(prompt=full_prompt)
        return jsonify({"reply": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)