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
        # Read and initialize the symbolic.metta file
        with open('symbolic.metta', 'r') as file:
            kb_init = file.read()
            
        # Initialize the knowledge base with all the rules and functions
        metta.run(kb_init)
        logger.info("MeTTa knowledge base initialized from symbolic.metta")
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

def convert_to_serializable(obj):
    """Convert MeTTa objects to JSON-serializable format."""
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        return [convert_to_serializable(item) for item in obj]
    if hasattr(obj, 'get_children'):  # ExpressionAtom
        return str(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    return str(obj)

def query_metta(symptoms, patient):
    try:
        symbolic_steps = []
        
        # Clear previous facts and reinitialize knowledge base
        metta.run('!(bind! &kb (new-space))')
        init_metta()  # Reload all rules and functions
        
        # Add symptoms as facts in the same format as symbolic.metta
        for i, symptom in enumerate(symptoms, start=1):
            fact = f'!(add-atom &kb (: FACT{i} (Evaluation {symptom} {patient})))'
            logger.info(f"Adding fact: {fact}")  # Add logging to debug fact formation
            metta.run(fact)
            symbolic_steps.append({
                'step': 'Adding Symptom',
                'fact': f'FACT{i}: {symptom} for {patient}'
            })
        
        symbolic_steps.append({'step': 'Starting Diagnosis Query', 'details': 'Checking for conditions using symbolic rules'})
        
        # Run forward chaining from first symptom, exactly as in symbolic.metta
        chain_query = f'!(fcc &kb (fromNumber 4) (: FACT1 (Evaluation {symptoms[0]} {patient})))'
        logger.info(f"Running chain query: {chain_query}")  # Add logging for chain query
        result = metta.run(chain_query)
        logger.info(f"Forward chaining result: {result}")
        
        diagnoses = []
        if result:
            # Process the raw MeTTa output
            raw_output = convert_to_serializable(result)
            symbolic_steps.append({
                'step': 'Raw Reasoning Chain',
                'result': raw_output
            })
            
            # Extract diagnoses from the reasoning chain
            for chain_result in result:
                if isinstance(chain_result, str) and '(Inheritance ' in chain_result:
                    # Extract condition from the result
                    condition = str(chain_result).split(' ')[-1].rstrip(')')
                    if condition not in diagnoses:
                        diagnoses.append(condition)
                        symbolic_steps.append({
                            'step': 'Found Condition',
                            'result': f'Diagnosed condition: {condition}'
                        })
        
        if diagnoses:
            symbolic_steps.append({
                'step': 'Conditions Found',
                'result': diagnoses
            })
            
            # Look up treatments using the rules from symbolic.metta
            for diagnosis in diagnoses:
                treatment_query = f'!(match &kb (-> (Inheritance {patient} {diagnosis}) (Evaluation $treatment {patient})) $treatment)'
                treatment = metta.run(treatment_query)
                if treatment:
                    symbolic_steps.append({
                        'step': 'Treatment Found',
                        'condition': diagnosis,
                        'treatment': convert_to_serializable(treatment)
                    })
        else:
            symbolic_steps.append({
                'step': 'Diagnosis Result',
                'details': 'No matching conditions found in knowledge base'
            })
        
        return {
            'symbolic_steps': symbolic_steps,
            'diagnoses': diagnoses,
            'raw_result': convert_to_serializable(result)  # Include raw MeTTa output
        }
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

        # Extract symptoms using GPT-4
        symptom_extraction_prompt = f"""
        Extract medical symptoms from this message: "{user_message}"
        Return only the symptoms in the format: has_fever, has_cough, etc.
        Use underscores between words and prefix with 'has_'.
        If no clear symptoms are mentioned, return an empty list.
        Format each symptom exactly as it appears in medical terminology (e.g., has_shortness_of_breath for breathing difficulty).
        """
        
        user_proxy.initiate_chat(
            assistant,
            message=symptom_extraction_prompt,
            silent=True
        )
        
        symptoms_text = user_proxy.last_response
        symptoms = [s.strip() for s in symptoms_text.split(',') if s.strip()]
        
        symbolic_results = None
        if symptoms:
            # Get MeTTa symbolic reasoning results
            symbolic_results = query_metta(symptoms, "current_patient")
            
            # Format symbolic reasoning steps for display
            symbolic_output = "Symbolic Reasoning Steps:\n"
            
            # Show the symptoms we're working with
            symbolic_output += "\nIdentified Symptoms:\n"
            for symptom in symptoms:
                symbolic_output += f"  • {symptom}\n"
            
            # Show the reasoning steps
            symbolic_output += "\nReasoning Process:\n"
            for step in symbolic_results['symbolic_steps']:
                symbolic_output += f"\n• {step['step']}:\n"
                for key, value in step.items():
                    if key != 'step':
                        symbolic_output += f"  - {key}: {value}\n"
            
            # Get GPT-4 analysis
            analysis_prompt = f"""
            Patient message: {user_message}
            Extracted symptoms: {symptoms}
            
            Symbolic reasoning results:
            {symbolic_output}
            
            Please provide a clear diagnosis analysis that:
            1. Acknowledges the symbolic reasoning results above
            2. Explains which rules were triggered and why
            3. Provides additional insights beyond the rule-based system
            4. Suggests a treatment plan
            
            Format your response with clear sections:
            - Symbolic System Findings:
            - Additional Neural Analysis:
            - Treatment Recommendations:
            """
            
            user_proxy.initiate_chat(
                assistant,
                message=analysis_prompt,
                silent=True
            )
            
            response = {
                'symbolic_reasoning': symbolic_output,
                'neural_analysis': user_proxy.last_response,
                'raw_symbolic_data': convert_to_serializable(symbolic_results)
            }
        else:
            response = {
                'error': "No symptoms identified",
                'message': "I couldn't identify any specific medical symptoms in your message. Could you please describe your symptoms more clearly?"
            }

        # Update chat history with structured response
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