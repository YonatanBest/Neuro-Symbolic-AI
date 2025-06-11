from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from hyperon import MeTTa
import autogen
import openai
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app)

# Configure more detailed logging for Flask
app.logger.setLevel(logging.INFO)

metta = MeTTa()

# Initialize MeTTa knowledge base
def init_metta():
    try:
        # Read the symbolic.metta file
        with open('symbolic.metta', 'r') as file:
            kb_init = file.read()
        
        # Initialize the knowledge base
        metta.run(kb_init)
        logger.info("MeTTa knowledge base initialized successfully from symbolic.metta")
    except Exception as e:
        logger.error(f"Error initializing MeTTa knowledge base: {str(e)}")
        raise

# Initialize GPT-4 config
config_list = [
    {
        'model': 'gpt-4',
        'api_key': os.getenv('OPENAI_API_KEY'),
    }
]

if not os.getenv('OPENAI_API_KEY'):
    logger.warning("OPENAI_API_KEY not found in environment variables")

# Create AutoGen agents with Docker disabled
assistant = autogen.AssistantAgent(
    name="doctor",
    llm_config={
        "config_list": config_list,
        "temperature": 0.7,
        "max_tokens": 500,
    },
    system_message="""You are a medical expert who helps diagnose conditions based on symptoms.
    Analyze the patient's symptoms and provide a clear diagnosis.
    Consider both the symbolic reasoning results from MeTTa and the symptoms provided.
    Format your response in a clear, structured way.
    Respond directly without asking follow-up questions."""
)

class WebUserProxyAgent(autogen.UserProxyAgent):
    def __init__(self):
        super().__init__(
            name="patient",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=1,  # Prevent conversation loops
            code_execution_config=False  # Disable code execution completely
        )
        self.last_response = None
    
    def receive(self, message, sender, request_reply=False, silent=False):
        if sender.name == "doctor":
            content = message.get('content', str(message)) if isinstance(message, dict) else str(message)
            self.last_response = content
            logger.info(f"Received response from doctor: {content[:100]}...")
            return False  # Don't continue the conversation
        return super().receive(message, sender, request_reply, silent)

user_proxy = WebUserProxyAgent()

def query_metta(symptoms, patient):
    try:
        # Add symptoms to MeTTa KB
        for i, symptom in enumerate(symptoms, start=1):
            logger.info(f"Adding symptom to KB: {symptom} for patient {patient}")
            metta.run(f'!(add-atom &kb (: FACT{i} (Evaluation {symptom} {patient})))')
        
        # Query for diagnosis
        logger.info(f"Querying MeTTa for diagnosis with first symptom: {symptoms[0]}")
        result = metta.run(f'!(fcc &kb (fromNumber 5) (: FACT1 (Evaluation {symptoms[0]} {patient})))')
        logger.info(f"MeTTa diagnosis result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in query_metta: {str(e)}")
        raise

def get_treatment(diagnosis, patient):
    try:
        logger.info(f"Getting treatment recommendation for patient {patient}")
        result = metta.run(f'!(match &kb (Evaluation recommend_$treatment {patient}) $treatment)')
        logger.info(f"Treatment result: {result}")
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error in get_treatment: {str(e)}")
        raise

@app.route('/')
def home():
    if 'chat_history' not in session:
        session['chat_history'] = []
    return render_template('index.html', chat_history=session['chat_history'])

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        logger.info(f"Received chat message: {user_message}")

        # Extract symptoms from user message using GPT-4
        symptom_extraction_prompt = f"""
        Extract medical symptoms from this message: "{user_message}"
        Return only the symptoms in the format: has_fever, has_cough, etc.
        If no clear symptoms are mentioned, return an empty list.
        """
        
        user_proxy.initiate_chat(
            assistant,
            message=symptom_extraction_prompt,
            silent=True
        )
        
        symptoms_text = user_proxy.last_response
        symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]
        
        if symptoms:
            # Get MeTTa diagnosis
            metta_diagnosis = query_metta(symptoms, "current_patient")
            
            # Get GPT-4 analysis
            analysis_prompt = f"""
            Patient message: {user_message}
            Extracted symptoms: {symptoms}
            MeTTa symbolic reasoning result: {metta_diagnosis}
            
            Please provide a clear, human-readable diagnosis and recommendations.
            """
            
            user_proxy.initiate_chat(
                assistant,
                message=analysis_prompt,
                silent=True
            )
            
            response = user_proxy.last_response
        else:
            response = "I couldn't identify any specific medical symptoms in your message. Could you please describe your symptoms more clearly?"

        # Update chat history
        chat_entry = {
            'user_message': user_message,
            'ai_response': response
        }
        
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        session['chat_history'].append(chat_entry)
        session.modified = True

        return jsonify({
            'response': response,
            'chat_history': session['chat_history']
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return jsonify({
            'error': 'An error occurred during the chat',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    init_metta()
    app.run(debug=True) 