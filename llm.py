import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model and tokenizer
# Consider adding error handling for model loading
try:
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    # Depending on the application, you might want to exit or handle this differently
    tokenizer = None
    model = None


# Load mood-question mapping - Add error handling
try:
    with open("moods1.json", "r", encoding="utf-8") as f:
        mood_questions = json.load(f)
except FileNotFoundError:
    print("Warning: moods1.json not found. Mood-based features might not work.")
    mood_questions = []
except json.JSONDecodeError:
    print("Warning: Error decoding moods1.json. Mood-based features might not work.")
    mood_questions = []
except Exception as e:
    print(f"Error loading moods1.json: {e}")
    mood_questions = []


# List of valid moods
valid_moods = [
    "Energized", "Focused", "Happy", "Collaborative", "Confident",
    "Neutral", "Tired", "Calm", "Stressed", "Frustrated",
    "Anxious", "Bored", "Overwhelmed", "Lonely", "Burned Out"
]

face_mood = "Anxious"

# --- NEW FUNCTION for direct chat ---
def generate_chat_response(user_input):
    """Generates a conversational response focused on mental wellness."""
    if not model or not tokenizer:
        print("Error: Model or tokenizer not loaded.")
        return "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment."

    # Enhanced prompt with mental wellness focus
    prompt = f"""The following is a supportive conversation with an AI mental wellness companion named Cognitive Harmony. 
The AI is empathetic, warm, and focused on helping the user improve their mental wellbeing.
The AI listens carefully and responds thoughtfully, without giving medical advice.

Human: {user_input}
AI:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)

        # Adjust generation parameters as needed
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,  # Limit response length
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id, # Ensure generation stops appropriately
            do_sample=True, # Use sampling for more varied responses
            temperature=0.7,
            top_p=0.9
        )
        # Decode the full output
        full_decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the AI's response part after the last "AI:"
        response = full_decoded_output.split("AI:")[-1].strip()

        # Basic safety check / filtering (optional but recommended)
        if not response:
             return "I'm not sure how to respond to that. Could you rephrase?"

        return response

    except Exception as e:
        print(f"[Chat Generation Error] {e}")
        # Fallback response
        return "I'm having trouble formulating a response right now. Please try again in a moment."

# --- Existing Functions (Keep them if still needed for other parts) ---

def detect_mood(user_input):
    """Detect user's mood based on input using Phi-2."""
    if not model or not tokenizer: return "Neutral"
    prompt = f"""Given the following text, what is the user's most likely mood? Choose only one from this list: {', '.join(valid_moods)}. Text: {user_input}
Mood:"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    try:
        outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # More robust extraction
        mood_part = decoded_output.split("Mood:")[-1].strip()
        # Find the first word in the mood part that is a valid mood
        detected_mood = "Neutral" # Default
        for word in mood_part.split():
             cleaned_word = ''.join(filter(str.isalpha, word)) # Remove punctuation if any
             if cleaned_word in valid_moods:
                  detected_mood = cleaned_word
                  break
        print(f"Detected mood raw: {mood_part}, Mapped: {detected_mood}")
        return detected_mood # No need to map again if we constrain the output
    except Exception as e:
        print(f"[Mood Detection Error] {e} — defaulting to 'Neutral'")
        return "Neutral"

# Map to valid mood might not be needed if detection prompt is good, but keep for safety
def map_to_valid_mood(mood):
    """Map model output to nearest valid mood using cosine similarity."""
    if mood in valid_moods:
        return mood
    if not model or not tokenizer: return "Neutral"

    # Basic string matching fallback
    mood_lower = mood.lower()
    for valid in valid_moods:
        if valid.lower() in mood_lower:
            print(f"Mapped '{mood}' to '{valid}' via string matching.")
            return valid

    # Embedding similarity (keep as more advanced fallback)
    try:
        # Ensure embeddings are calculated correctly
        with torch.no_grad(): # Disable gradient calculations
             valid_mood_tokens = tokenizer(valid_moods, return_tensors="pt", padding=True, truncation=True).to(model.device)
             mood_embeddings = model.get_input_embeddings()(valid_mood_tokens.input_ids).mean(dim=1).cpu().numpy()

             input_tokens = tokenizer(mood, return_tensors="pt", truncation=True).to(model.device)
             input_embedding = model.get_input_embeddings()(input_tokens.input_ids).mean(dim=1).cpu().numpy()

        if input_embedding.shape[0] == 0 or mood_embeddings.shape[0] == 0:
             raise ValueError("Embedding calculation resulted in empty tensor.")

        similarities = cosine_similarity(input_embedding.reshape(1, -1), mood_embeddings).flatten()
        closest_mood_index = int(np.argmax(similarities))
        mapped = valid_moods[closest_mood_index]
        print(f"Mapped '{mood}' to '{mapped}' via embedding similarity.")
        return mapped
    except Exception as e:
        print(f"[Mood Mapping Error] {e} — returning 'Neutral'")
        return "Neutral"


def personalize_questions(mood, user_input):
    """Use Phi-2 to personalize up to 3 questions based on mood and user input."""
    if not model or not tokenizer or not mood_questions: return ["How can I help you further today?"] # Fallback

    matching_questions = [entry for entry in mood_questions if entry.get("mood", "").lower() == mood.lower()]
    if not matching_questions:
        return ["How are you feeling overall right now?", "Is there anything specific you'd like to talk about?"]

    # Limit the number of questions to personalize
    selected = matching_questions[:3] # Max 3 questions
    personalized_questions = []

    for entry in selected:
        base_question = entry["question"]
        tags = ", ".join(entry.get("tags", []))
        prompt = f"""User input: '{user_input}'
User mood: {mood}
Context Tags: {tags}
Original Question: '{base_question}'
Based on the user's input and mood, refine or personalize the original question to make it more relevant and empathetic. If the original question is already suitable, you can use it directly.
Personalized Question:"""

        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)

        try:
            outputs = model.generate(
                 **inputs,
                 max_new_tokens=70, # Shorter length for questions
                 pad_token_id=tokenizer.eos_token_id,
                 eos_token_id=tokenizer.eos_token_id,
                 do_sample=True,
                 temperature=0.6, # Slightly lower temp for more focused questions
                 top_p=0.9
            )
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the text after the last occurrence of "Personalized Question:"
            personalized = decoded.split("Personalized Question:")[-1].strip()
            # Simple validation: ensure it's a question
            if "?" not in personalized:
                 personalized = base_question # Fallback if generation fails badly
            personalized_questions.append(personalized)
        except Exception as e:
            print(f"[Question Personalization Error] {e} — using original question")
            personalized_questions.append(base_question) # Fallback to original

    return personalized_questions


def generate_final_message(user_input, text_mood, face_mood, answers):
    """Generate a summary message using the model."""
    if not model or not tokenizer: return "Thank you for sharing."
    answers_str = "\n".join([f"- {a}" for a in answers if a]) # Ensure answers are not empty
    prompt = f"""A user provided the following input: '{user_input}'.
Their detected text mood was '{text_mood}' and detected face mood was '{face_mood}'.
They answered the following questions:
{answers_str}

Based on all this information, provide a brief, empathetic, and supportive summary message for the user:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)
        outputs = model.generate(
             **inputs,
             max_new_tokens=150,
             pad_token_id=tokenizer.eos_token_id,
             eos_token_id=tokenizer.eos_token_id,
             do_sample=True,
             temperature=0.7,
             top_p=0.9
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract text after the prompt
        summary = decoded[len(prompt):].strip() # Get text generated after the prompt
        if not summary: # Basic fallback
             summary = "Thank you for sharing your thoughts and feelings. Remember to be kind to yourself."
        return summary
    except Exception as e:
        print(f"[Final Message Generation Error] {e}")
        return "We understand you're going through something. We're here to support you."


def generate_api_prompt(user_input, text_mood, face_mood, answers):
    """Generate a prompt suitable for external AI support."""
    if not model or not tokenizer: return "Provide empathetic support." # Minimal fallback
    answers_str = "\n".join([f"- {a}" for a in answers if a])
    prompt = f"""User's initial input: '{user_input}'.
Detected text mood: '{text_mood}'.
Detected face mood: '{face_mood}'.
User's answers to questions:
{answers_str}

Generate a concise prompt for an empathetic AI assistant to offer support based ONLY on the information provided above:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False # More deterministic for API prompt
        )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        api_prompt_text = decoded[len(prompt):].strip() # Get text after prompt
        if not api_prompt_text: # Basic fallback
             api_prompt_text = f"User is feeling {text_mood}. Input: {user_input}. Please respond empathetically."
        return api_prompt_text
    except Exception as e:
        print(f"[API Prompt Generation Error] {e}")
        return "Please be empathetic and supportive based on the user's emotional state."