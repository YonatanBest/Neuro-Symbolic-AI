# Neuro-Symbolic AI Medical Diagnosis System

A sophisticated medical diagnosis system that combines symbolic reasoning (MeTTa) with neural processing (GPT-4) to provide comprehensive medical assessments and recommendations.

## Features

- **Hybrid Architecture**
  - Symbolic reasoning using MeTTa for rule-based diagnosis
  - Neural processing using GPT-4 for natural language understanding and response generation
  - Seamless integration of both approaches for accurate medical assessments

- **Comprehensive Disease Coverage**
  - Multiple conditions including asthma, flu, COVID-19, diabetes, and more
  - Detailed symptom-to-diagnosis reasoning chains
  - Evidence-based treatment recommendations

- **Treatment Recommendations**
  - Four-tier treatment framework:
    * Primary treatment approaches
    * Medication recommendations
    * Lifestyle modifications
    * Monitoring requirements
  - Emergency care guidelines
  - Personalized care suggestions

- **User-Friendly Interface**
  - Clean, modern web interface
  - Real-time interaction
  - Structured, easy-to-understand responses
  - Mobile-responsive design

## Technical Stack

- **Backend**
  - Python Flask server
  - MeTTa for symbolic reasoning
  - OpenAI GPT-4 for natural language processing
  - AutoGen for agent-based interactions

- **Frontend**
  - HTML5/CSS3
  - JavaScript
  - jQuery
  - Responsive design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/YonatanBest/Neuro-Symbolic-AI.git
cd Neuro-Symbolic-AI
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
# Create a .env file with your OpenAI API key
echo "OPENAI_API_KEY=your_api_key_here" > .env
```

4. Run the application:
```bash
python app.py
```

5. Access the web interface at `http://localhost:5000`

## System Architecture

### Symbolic Reasoning (MeTTa)
- Rule-based knowledge representation
- Forward chaining for diagnosis
- Structured treatment recommendations
- Logical inference engine

### Neural Processing (GPT-4)
- Natural language understanding
- Symptom extraction
- Contextual analysis
- Human-like response generation

### Integration Layer
- Seamless combination of symbolic and neural outputs
- Structured response formatting
- Confidence scoring
- Treatment prioritization

## Usage

1. Enter symptoms in natural language
2. System processes input through:
   - GPT-4 for symptom extraction
   - MeTTa for symbolic reasoning
   - Combined analysis for final diagnosis
3. Receive structured output with:
   - Identified condition
   - Detailed explanation
   - Treatment recommendations
   - Important precautions

## Response Format

The system provides structured responses with:

1. **Disease Information**
   - Condition name
   - Confidence level
   - Diagnostic reasoning

2. **Detailed Information**
   - Condition description
   - Symptom correlation
   - Causes and risk factors
   - Expected progression

3. **Treatment Plan**
   - Immediate actions
   - Medication recommendations
   - Lifestyle modifications
   - Monitoring requirements
   - Emergency care guidelines

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This system is designed as a supplementary tool and should not replace professional medical advice. Always consult with healthcare professionals for proper diagnosis and treatment.

## Acknowledgments

- MeTTa team for the symbolic reasoning engine
- OpenAI for GPT-4 API
- AutoGen team for the agent framework
- All contributors and testers 