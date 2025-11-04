"""
LLM-based Sentence Generator for Sign Language Translation
========================================================

This module uses a local LLM (via Ollama) to convert sign language words into meaningful sentences.
Works completely offline with no API keys required.
"""

import requests
import json
import time
from typing import List, Optional

class LLMSentenceGenerator:
    def __init__(self, model_name: str = "llama3.2:1b", base_url: str = "http://localhost:11434"):
        """
        Initialize the LLM sentence generator.
        
        Args:
            model_name: The Ollama model to use (default: llama3.2:1b - lightweight)
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"
        self.is_available = self._check_ollama_availability()
        
        if not self.is_available:
            print("⚠️ Ollama not available. Install Ollama and pull a model first.")
            print("Commands to run:")
            print("1. Install Ollama: https://ollama.ai/")
            print(f"2. Pull model: ollama pull {model_name}")
            print("3. Start Ollama: ollama serve")
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                if self.model_name in model_names:
                    print(f"✅ Ollama is running with model: {self.model_name}")
                    return True
                else:
                    print(f"⚠️ Model {self.model_name} not found. Available models: {model_names}")
                    return False
            return False
        except Exception as e:
            print(f"❌ Ollama not available: {e}")
            return False
    
    def generate_sentence(self, words: List[str], context: str = "") -> str:
        """
        Generate a meaningful sentence from sign language words using LLM.
        
        Args:
            words: List of predicted sign language words
            context: Optional conversation context
            
        Returns:
            Generated meaningful sentence
        """
        if not words:
            return ""
        
        if not self.is_available:
            return " ".join(words).capitalize() + "."  # Fallback to simple concatenation
        
        # Create prompt for the LLM
        prompt = self._create_prompt(words, context)

        response = self._call_llm(prompt)
        cleaned_response = self._clean_response(response)

        return cleaned_response


    
    def _create_prompt(self, words: List[str], context: str = "") -> str:
        """Create a very strict prompt for the LLM."""
        words_str = " ".join(words)
        
        prompt = f"""Arrange these words into a sentence: {words_str}

CRITICAL RULES:
- Use EXACTLY these words: {words_str}
- Do NOT add any other words
- Do NOT change the words
- Do NOT add explanations
- Just arrange them into a sentence
- Output only the sentence

Sentence:"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,  # Very low temperature for strict responses
                "top_p": 0.3,        # Very low top_p for focused responses
                "max_tokens": 50,     # Limit tokens to prevent long responses
                "repeat_penalty": 1.1,
                "stop": ["\n", ".", "!", "?"]  # Stop at sentence endings
            }
        }
        
        response = requests.post(self.api_url, json=payload, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        return result.get('response', '')
    
    def _clean_response(self, response: str) -> str:
        """Clean and format the LLM response aggressively."""
        response = response.strip()
        
        # Remove common LLM artifacts
        artifacts = [
            "Here's the sentence:",
            "The sentence is:",
            "Sentence:",
            "Here's:",
            "The converted sentence:",
            "Converted:",
            "Output:",
            "Result:",
            "Answer:",
            "Here is the sentence:",
            "The sentence would be:",
            "A sentence using these words:"
        ]
        
        for artifact in artifacts:
            if response.lower().startswith(artifact.lower()):
                response = response[len(artifact):].strip()
                break
        
        # Remove any text before the first sentence
        sentences = response.split('.')
        if sentences:
            response = sentences[0].strip()
            if response:
                response += "."
        
        # Remove any text after the first sentence
        if '\n' in response:
            response = response.split('\n')[0].strip()
        
        # Ensure proper capitalization
        if response:
            response = response[0].upper() + response[1:]
        
        # Ensure proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += "."
        
        return response
    
    def generate_multiple_sentences(self, words: List[str], context: str = "") -> List[str]:
        """Generate multiple possible sentences from the words."""
        if not words:
            return []
        
        sentences = []
        
        # Generate different variations
        variations = [
            f"Convert these sign language words into a question: {' '.join(words)}",
            f"Convert these sign language words into a statement: {' '.join(words)}",
            f"Convert these sign language words into a greeting: {' '.join(words)}"
        ]
        
        for variation in variations:
            try:
                if self.is_available:
                    response = self._call_llm(variation)
                    sentence = self._clean_response(response)
                    if sentence and sentence not in sentences:
                        sentences.append(sentence)
                else:
                    # Fallback variations
                    if "question" in variation.lower():
                        sentence = " ".join(words).capitalize() + "?"
                    elif "greeting" in variation.lower():
                        sentence = "Hello, " + " ".join(words).lower() + "!"
                    else:
                        sentence = " ".join(words).capitalize() + "."
                    
                    if sentence not in sentences:
                        sentences.append(sentence)
            except Exception as e:
                print(f"❌ Error generating variation: {e}")
                continue
        
        return sentences[:3]  # Return top 3
    
    def update_context(self, sentence: str):
        """Update conversation context for better sentence generation."""
        # This could be used to maintain conversation history
        pass

def install_ollama_instructions():
    """Print instructions for installing and setting up Ollama."""
    print("""
🚀 To use LLM-based sentence generation, you need to install Ollama:

1. Install Ollama:
   - Windows: Download from https://ollama.ai/
   - Or run: winget install Ollama.Ollama

2. Pull a lightweight model:
   ollama pull llama3.2:1b

3. Start Ollama server:
   ollama serve

4. Test the setup:
   python test_llm_generator.py
""")

if __name__ == "__main__":
    # Test the LLM generator
    generator = LLMSentenceGenerator()
    
    if not generator.is_available:
        install_ollama_instructions()
    else:
        # Test cases
        test_cases = [
            ["what", "your", "name"],
            ["I", "am", "happy"],
            ["where", "is", "home"],
            ["hello", "how", "are", "you"],
            ["thank", "you", "very", "much"]
        ]
        
        print("Testing LLM Sentence Generator:")
        print("=" * 50)
        
        for words in test_cases:
            sentence = generator.generate_sentence(words)
            print(f"Input: {words}")
            print(f"Output: {sentence}")
            print()
