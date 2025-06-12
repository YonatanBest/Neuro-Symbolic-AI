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
    system_message="""You are a knowledgeable AI assistant with medical expertise. Your primary role is to help diagnose conditions based on symptoms, but you can also engage in general conversation and answer non-medical questions.

    For medical queries:
    - Analyze symptoms and provide clear diagnoses
    - Consider both symbolic reasoning results from MeTTa and the symptoms provided
    - Format medical responses in a clear, structured way
    - Include appropriate medical disclaimers

    For non-medical queries:
    - Respond naturally and informatively
    - Draw from your general knowledge
    - Maintain a helpful and professional tone
    - Make it clear that you're stepping outside the medical domain

    Always:
    - Be direct and concise
    - Stay within your knowledge boundaries
    - Maintain a professional yet friendly tone
    """
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
            logger.info(f"Adding fact: {fact}")
            metta.run(fact)
            symbolic_steps.append({
                'step': 'Adding Symptom',
                'fact': f'FACT{i}: {symptom} for {patient}'
            })
        
        symbolic_steps.append({'step': 'Starting Diagnosis Query', 'details': 'Checking for conditions using symbolic rules'})
        
        # Run forward chaining from first symptom
        chain_query = f'!(fcc &kb (fromNumber 4) (: FACT1 (Evaluation {symptoms[0]} {patient})))'
        logger.info(f"Running chain query: {chain_query}")
        result = metta.run(chain_query)
        logger.info(f"Forward chaining result: {result}")
        
        if result:
            # Process the raw MeTTa output
            raw_output = convert_to_serializable(result)
            
            # Run additional chains for treatments if a diagnosis was found
            if any('(Inheritance ' in str(r) for r in result):
                # Query for treatments
                treatment_result = metta.run(f'!(match &kb (Treatment {patient} $treatments) $treatments)')
                if treatment_result:
                    raw_output.extend(convert_to_serializable(treatment_result))
                
                # Query for medications
                medication_result = metta.run(f'!(match &kb (Medication {patient} $medications) $medications)')
                if medication_result:
                    raw_output.extend(convert_to_serializable(medication_result))
                
                # Query for lifestyle recommendations
                lifestyle_result = metta.run(f'!(match &kb (Lifestyle {patient} $lifestyle) $lifestyle)')
                if lifestyle_result:
                    raw_output.extend(convert_to_serializable(lifestyle_result))
                
                # Query for monitoring recommendations
                monitoring_result = metta.run(f'!(match &kb (Monitoring {patient} $monitoring) $monitoring)')
                if monitoring_result:
                    raw_output.extend(convert_to_serializable(monitoring_result))
            
            symbolic_steps.append({
                'step': 'Raw Reasoning Chain',
                'result': raw_output
            })
            
            # Run additional chains if needed
            for i, symptom in enumerate(symptoms[1:], start=2):
                chain_query = f'!(fcc &kb (fromNumber 4) (: FACT{i} (Evaluation {symptom} {patient})))'
                chain_result = metta.run(chain_query)
                if chain_result:
                    raw_output.extend(convert_to_serializable(chain_result))
        
        return {
            'symbolic_steps': symbolic_steps,
            'raw_result': convert_to_serializable(result)
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

        # First, determine if this is a medical query
        query_type_prompt = f"""
        Analyze if this message is asking for medical advice or diagnosis: "{user_message}"
        Return ONLY 'medical' or 'non_medical'.
        Consider it medical if it mentions:
        - Symptoms
        - Health conditions
        - Medical treatments
        - Health concerns
        Otherwise, return non_medical.
        """

        user_proxy.initiate_chat(
            assistant,
            message=query_type_prompt,
            silent=True
        )
        
        query_type = user_proxy.last_response.strip().lower()

        if query_type == 'medical':
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
                
                # Show the symptoms we're working with
                symbolic_output = "\nIdentified Symptoms:\n"
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
                You are a friendly and knowledgeable doctor explaining the results of a medical diagnosis system to a patient.
                
                The patient reported these symptoms: {user_message}

                Our medical reasoning system analyzed these symptoms and here's what it found:

                {symbolic_output}

                The reasoning chain shows:
                {symbolic_results['raw_result']}

                Please provide your analysis in the following structured format:

                Disease:
                - Name of the identified condition
                - Confidence level in the diagnosis

                More Information:
                - Brief description of the condition
                - How the symptoms relate to the diagnosis
                - Common causes and risk factors
                - Typical duration and progression

                Treatment:
                - Immediate steps to take
                - Recommended medications
                - Lifestyle changes
                - Monitoring requirements
                - When to seek emergency care

                Important Note:
                While our AI system can help identify possible conditions, this is not a substitute for professional medical care. Please consult with a healthcare provider for proper diagnosis and treatment.

                Use a warm, caring tone and avoid technical jargon. Break down the medical reasoning in a way that's easy to understand."""
                
                user_proxy.initiate_chat(
                    assistant,
                    message=analysis_prompt,
                    silent=True
                )
                
                response = {
                    'type': 'medical',
                    'symbolic_reasoning': symbolic_output,
                    'neural_analysis': user_proxy.last_response,
                    'raw_symbolic_data': symbolic_results['raw_result']
                }
            else:
                response = {
                    'type': 'medical',
                    'error': "No symptoms identified",
                    'message': "I couldn't identify any specific medical symptoms in your message. Could you please describe your symptoms more clearly?"
                }
        else:
            # Handle non-medical query
            general_prompt = f"""
            The user has asked a non-medical question: "{user_message}"
            
            Please provide a helpful response while:
            1. Making it clear you're answering as a general AI assistant
            2. Drawing from your general knowledge to provide accurate information
            3. Maintaining a professional yet conversational tone
            4. Staying within the bounds of your knowledge and capabilities
            
            If the query is unclear or too broad, ask for clarification."""

            user_proxy.initiate_chat(
                assistant,
                message=general_prompt,
                silent=True
            )

            response = {
                'type': 'non_medical',
                'response': user_proxy.last_response
            }

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