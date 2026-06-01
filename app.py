from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import requests
import os
from werkzeug.utils import secure_filename  
from predict_disease import predict_disease
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
from utils.translator import translate_text




translator = Translator()

def translate_global(text):
    lang = session.get("lang", "en")

    if lang == "en":
        return text

    try:
        return translator.translate(text, dest=lang).text
    except:
        return text

df_chatbot = pd.read_csv("data/agriculture_chatbot_500.csv")

chat_questions = df_chatbot["question"].tolist()
chat_answers = df_chatbot["answer"].tolist()

chat_model = SentenceTransformer('all-MiniLM-L6-v2')
chat_embeddings = chat_model.encode(chat_questions)

SUPPORTED_LANGUAGES = ["en", "ml", "ta", "te", "kn", "hi"]

def auto_translate(data, lang):
    if lang == "en":
        return data

    if isinstance(data, dict):
        return {k: auto_translate(v, lang) for k, v in data.items()}

    if isinstance(data, list):
        return [auto_translate(i, lang) for i in data]

    if isinstance(data, str):
        try:
            return translator.translate(data, dest=lang).text
        except:
            return data

    return data

# ---------------- LANGUAGE DICTIONARY ----------------
LANGUAGES = {

    "en": {
        "login": "Login",
        "register": "Register",
        "email": "Email",
        "phone": "Phone",
        "name": "Name",
        "state": "State",
        "dashboard": "Dashboard",
        "subtitle": "Your Smart Farming Companion",

        "disease": "Disease Detector",
        "disease_desc": "Scan crops or upload images to detect diseases.",
        "disease_btn": "Scan / Upload",

        "schemes": "Government Schemes",
        "schemes_desc": "Central & State agriculture schemes.",
        "schemes_btn": "View Schemes",

        "market": "Market Near Me",
        "market_desc": "Find nearby markets to sell crops.",
        "market_btn": "Find Markets",

        "prices": "Market Prices",
        "prices_desc": "Check current crop prices.",
        "prices_btn": "View Prices",

        "search_crop": "Search crop...",
        "crop_not_found": "❌ Crop not found",
        "price": "Price",

        "weather": "Weather Forecast",
        "weather_desc": "Get live weather updates.",
        "weather_btn": "Enable Location",

        "soil": "Soil Information",
        "soil_desc": "Explore soil advantages and crop suitability.",
        "soil_btn": "View Soil Info",

"calendar": "Farmer Calendar",
"select_state": "-- Select State --",
"loading_calendar": "Loading seasonal plan...",
"sowing": "Sowing",
"planting": "Planting",
"harvest": "Harvest",
"error_loading_calendar": "Error loading calendar",
"calendar_btn": "Open Calendar",

        "ai": "My Farmer (AI)",
        "ai_desc": "Friendly AI for agriculture guidance.",
        "ai_btn": "Chat Now",

        "schemes": "Government Schemes",
"central_schemes": "Central Government Schemes",
"select_state": "Select Your State",
"official_site": "Official Website",

"pmkisan_title": "Pradhan Mantri Kisan Samman Nidhi (PM-KISAN)",
"pmkisan_desc": "Provides ₹6000 per year income support to farmers.",

"pmfby_title": "Pradhan Mantri Fasal Bima Yojana (PMFBY)",
"pmfby_desc": "Insurance coverage for crop losses due to natural disasters.",

"soilhealth_title": "Soil Health Card Scheme",
"soilhealth_desc": "Provides soil testing and fertility recommendations.",

"weather_dashboard": "Weather Dashboard",
"enter_place": "Enter city / place",
"add_place": "Add Place",
"my_location": "My Location",
"latest_weather_news": "Latest Weather News",
"temp": "Temp",
"humidity": "Humidity",
"wind": "Wind",
"rain": "Rain",
"condition": "Condition",
"place_not_found": "Place not found",
"enter_valid_place": "Enter a valid place",

"soil": "Soil Information",
"soil_title": "Soil Information System",
"select_soil": "-- Select Soil Type --",
"loading_soil": "Loading soil insights...",
"advantages": "Advantages",
"limitations": "Limitations",
"best_crops": "Best Crops",
"fertilizer_tips": "Fertilizer Tips",
"pest_management": "Pest Management",
"error_loading_soil": "Error loading soil data"
    },

    "ml": {
        "login": "ലോഗിൻ",
        "register": "രജിസ്റ്റർ",
        "email": "ഇമെയിൽ",
        "phone": "ഫോൺ",
        "name": "പേര്",
        "state": "സംസ്ഥാനം",
        "dashboard": "ഡാഷ്ബോർഡ്",
        "subtitle": "നിങ്ങളുടെ സ്മാർട്ട് കൃഷി സഹായി",

        "disease": "രോഗം കണ്ടെത്തൽ",
        "disease_desc": "വിളകളിലെ രോഗങ്ങൾ കണ്ടെത്തുക.",
        "disease_btn": "സ്കാൻ ചെയ്യുക",

        "schemes": "സർക്കാർ പദ്ധതികൾ",
        "schemes_desc": "കേന്ദ്ര & സംസ്ഥാന കൃഷി പദ്ധതികൾ.",
        "schemes_btn": "പദ്ധതികൾ കാണുക",

        "market": "അടുത്തുള്ള മാർക്കറ്റ്",
        "market_desc": "വിളകൾ വിൽക്കാൻ മാർക്കറ്റുകൾ കണ്ടെത്തുക.",
        "market_btn": "മാർക്കറ്റുകൾ",

        "prices": "വിപണി വിലകൾ",
        "prices_desc": "ഇന്നത്തെ വിളവിലകൾ പരിശോധിക്കുക.",
        "prices_btn": "വില കാണുക",

        "search_crop": "വിള തിരയുക...",
        "crop_not_found": "❌ വിള കണ്ടെത്തിയില്ല",
        "price": "വില",

        "weather": "കാലാവസ്ഥ",
        "weather_desc": "തത്സമയ കാലാവസ്ഥ വിവരം.",
        "weather_btn": "ലൊക്കേഷൻ",

        "soil": "മണ്ണ് വിവരം",
        "soil_desc": "മണ്ണിന്റെ ഗുണങ്ങളും വിളകളും പരിശോധിക്കുക.",
        "soil_btn": "മണ്ണ് കാണുക",

        "calendar": "കർഷക കലണ്ടർ",
"select_state": "-- സംസ്ഥാനം തിരഞ്ഞെടുക്കുക --",
"loading_calendar": "സീസൺ പദ്ധതി ലോഡിംഗ്...",
"sowing": "വിത്തിടൽ",
"planting": "നട്ട് വളർത്തൽ",
"harvest": "വിളവെടുപ്പ്",
"error_loading_calendar": "കലണ്ടർ ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല",
"calendar_btn": "കലണ്ടർ തുറക്കുക",

        "ai": "എന്റെ ഫാർമർ AI",
        "ai_desc": "കൃഷിക്ക് സൗഹൃദ AI സഹായി.",
        "ai_btn": "ചാറ്റ്",

        "schemes": "സർക്കാർ പദ്ധതികൾ",
"central_schemes": "കേന്ദ്ര സർക്കാർ പദ്ധതികൾ",
"select_state": "നിങ്ങളുടെ സംസ്ഥാനം തിരഞ്ഞെടുക്കുക",
"official_site": "അധികൃത വെബ്സൈറ്റ്",

"pmkisan_title": "പ്രധാനമന്ത്രി കിസാൻ സമ്മാൻ നിധി (PM-KISAN)",
"pmkisan_desc": "കർഷകർക്ക് പ്രതിവർഷം ₹6,000 നേരിട്ടുള്ള വരുമാന സഹായം.",

"pmfby_title": "പ്രധാനമന്ത്രി ഫസൽ ബിമാ യോജന",
"pmfby_desc": "പ്രകൃതിദുരന്തങ്ങളാൽ വിളനഷ്ടം സംഭവിക്കുമ്പോൾ ഇൻഷുറൻസ് സംരക്ഷണം.",

"soilhealth_title": "സോയിൽ ഹെൽത്ത് കാർഡ് പദ്ധതി",
"soilhealth_desc": "മണ്ണ് പരിശോധനയും ഫലഭൂയിഷ്ഠത വിലയിരുത്തലും.",

"weather_dashboard": "കാലാവസ്ഥ ഡാഷ്ബോർഡ്",
"enter_place": "നഗരം / സ്ഥലം നൽകുക",
"add_place": "സ്ഥലം ചേർക്കുക",
"my_location": "എന്റെ ലൊക്കേഷൻ",
"latest_weather_news": "പുതിയ കാലാവസ്ഥ വാർത്തകൾ",
"temp": "താപനില",
"humidity": "ആർദ്രത",
"wind": "കാറ്റ്",
"rain": "മഴ",
"condition": "സ്ഥിതി",
"place_not_found": "സ്ഥലം കണ്ടെത്താനായില്ല",
"enter_valid_place": "ശരിയായ സ്ഥലം നൽകുക",

"soil": "മണ്ണ് വിവരം",
"soil_title": "മണ്ണ് വിവര സംവിധാനം",
"select_soil": "-- മണ്ണ് തരം തിരഞ്ഞെടുക്കുക --",
"loading_soil": "മണ്ണ് വിവരങ്ങൾ ലോഡിംഗ്...",
"advantages": "ഗുണങ്ങൾ",
"limitations": "പരിമിതികൾ",
"best_crops": "മികച്ച വിളകൾ",
"fertilizer_tips": "വള നിർദ്ദേശങ്ങൾ",
"pest_management": "കീട നിയന്ത്രണം",
"error_loading_soil": "മണ്ണ് ഡാറ്റ ലോഡ് ചെയ്യാൻ കഴിഞ്ഞില്ല"


    },

    "ta": {
        "login": "உள்நுழை",
        "register": "பதிவு",
        "email": "மின்னஞ்சல்",
        "phone": "தொலைபேசி",
        "name": "பெயர்",
        "state": "மாநிலம்",
        "dashboard": "டாஷ்போர்டு",
        "subtitle": "உங்கள் ஸ்மார்ட் விவசாய துணை",

        "disease": "நோய் கண்டறிதல்",
        "disease_desc": "பயிர் நோய்களை கண்டறியவும்.",
        "disease_btn": "ஸ்கேன்",

        "schemes": "அரசு திட்டங்கள்",
        "schemes_desc": "மத்திய & மாநில திட்டங்கள்.",
        "schemes_btn": "பார்க்க",

        "market": "அருகிலுள்ள சந்தை",
        "market_desc": "விற்க சந்தைகளை கண்டறியவும்.",
        "market_btn": "சந்தைகள்",

        "prices": "சந்தை விலைகள்",
        "prices_desc": "இன்றைய விலைகள்.",
        "prices_btn": "விலைகள்",

        "search_crop": "பயிர் தேடுக...",
        "crop_not_found": "❌ பயிர் கிடைக்கவில்லை",
        "price": "விலை",

        "weather": "வானிலை",
        "weather_desc": "நேரடி வானிலை தகவல்.",
        "weather_btn": "இடம்",

        "soil": "மண் தகவல்",
        "soil_desc": "மண் நன்மைகள் & பயிர்கள்.",
        "soil_btn": "மண் பார்க்க",
"calendar": "விவசாய காலண்டர்",
"select_state": "-- மாநிலத்தைத் தேர்ந்தெடுக்கவும் --",
"loading_calendar": "சீசன் திட்டம் ஏற்றப்படுகிறது...",
"sowing": "விதைப்பு",
"planting": "நட்டு வளர்த்தல்",
"harvest": "அறுவடை",
"error_loading_calendar": "காலண்டரை ஏற்ற முடியவில்லை",
"calendar_btn": "காலண்டர் திறக்க",
        "ai": "என் விவசாய AI",
        "ai_desc": "விவசாய வழிகாட்டி AI.",
        "ai_btn": "அரட்டை",

        "schemes": "அரசு திட்டங்கள்",
"central_schemes": "மத்திய அரசு திட்டங்கள்",
"select_state": "உங்கள் மாநிலத்தைத் தேர்ந்தெடுக்கவும்",
"official_site": "அதிகாரப்பூர்வ வலைத்தளம்",

"pmkisan_title": "பிரதான் மந்திரி கிசான் சம்மான் நிதி (PM-KISAN)",
"pmkisan_desc": "விவசாயிகளுக்கு வருடத்திற்கு ₹6,000 நேரடி வருமான ஆதரவு.",

"pmfby_title": "பிரதான் மந்திரி பயிர் காப்பீட்டு திட்டம்",
"pmfby_desc": "பயிர் இழப்புகளை பாதுகாக்க காப்பீட்டு திட்டம்.",

"soilhealth_title": "மண் சுகாதார அட்டை திட்டம்",
"soilhealth_desc": "மண் பரிசோதனை மற்றும் உர மதிப்பீடு.",

"weather_dashboard": "வானிலை டாஷ்போர்டு",
"enter_place": "நகரம் / இடம் உள்ளிடவும்",
"add_place": "இடம் சேர்க்க",
"my_location": "என் இருப்பிடம்",
"latest_weather_news": "சமீபத்திய வானிலை செய்திகள்",
"temp": "வெப்பநிலை",
"humidity": "ஈரப்பதம்",
"wind": "காற்று",
"rain": "மழை",
"condition": "நிலை",
"place_not_found": "இடம் கிடைக்கவில்லை",
"enter_valid_place": "சரியான இடம் உள்ளிடவும்",

"soil": "மண் தகவல்",
"soil_title": "மண் தகவல் அமைப்பு",
"select_soil": "-- மண் வகையைத் தேர்ந்தெடுக்கவும் --",
"loading_soil": "மண் தகவல் ஏற்றப்படுகிறது...",
"advantages": "நன்மைகள்",
"limitations": "குறைகள்",
"best_crops": "சிறந்த பயிர்கள்",
"fertilizer_tips": "உரம் ஆலோசனை",
"pest_management": "பூச்சி கட்டுப்பாடு",
"error_loading_soil": "மண் தரவை ஏற்ற முடியவில்லை"
    },

    "te": {
        "login": "లాగిన్",
        "register": "నమోదు",
        "email": "ఈమెయిల్",
        "phone": "ఫోన్",
        "name": "పేరు",
        "state": "రాష్ట్రం",
        "dashboard": "డాష్‌బోర్డ్",
        "subtitle": "మీ స్మార్ట్ వ్యవసాయ సహాయకుడు",

        "disease": "రోగ నిర్ధారణ",
        "disease_desc": "పంటల వ్యాధులను గుర్తించండి.",
        "disease_btn": "స్కాన్",

        "schemes": "ప్రభుత్వ పథకాలు",
        "schemes_desc": "కేంద్ర & రాష్ట్ర పథకాలు.",
        "schemes_btn": "చూడండి",

        "market": "సమీప మార్కెట్",
        "market_desc": "విక్రయానికి మార్కెట్లు.",
        "market_btn": "మార్కెట్లు",

        "prices": "మార్కెట్ ధరలు",
        "prices_desc": "నేటి ధరలు.",
        "prices_btn": "ధరలు",

        "search_crop": "పంట వెతకండి...",
        "crop_not_found": "❌ పంట కనబడలేదు",
        "price": "ధర",

        "weather": "వాతావరణం",
        "weather_desc": "ప్రత్యక్ష వాతావరణ సమాచారం.",
        "weather_btn": "లొకేషన్",

        "soil": "మట్టి సమాచారం",
        "soil_desc": "మట్టి ప్రయోజనాలు & పంటలు.",
        "soil_btn": "మట్టి చూడండి",

        "calendar": "రైతు క్యాలెండర్",
"select_state": "-- రాష్ట్రాన్ని ఎంచుకోండి --",
"loading_calendar": "సీజన్ ప్రణాళిక లోడ్ అవుతోంది...",
"sowing": "విత్తనం",
"planting": "నాటడం",
"harvest": "పంట కోత",
"error_loading_calendar": "క్యాలెండర్ లోడ్ చేయలేకపోయాం",
"calendar_btn": "క్యాలెండర్ తెరవండి",

        "ai": "నా రైతు AI",
        "ai_desc": "వ్యవసాయ మార్గదర్శక AI.",
        "ai_btn": "చాట్",

        "schemes": "ప్రభుత్వ పథకాలు",
"central_schemes": "కేంద్ర ప్రభుత్వం పథకాలు",
"select_state": "మీ రాష్ట్రాన్ని ఎంచుకోండి",
"official_site": "అధికారిక వెబ్‌సైట్",

"pmkisan_title": "ప్రధాన మంత్రి కిసాన్ సమ్మాన్ నిధి (PM-KISAN)",
"pmkisan_desc": "రైతులకు సంవత్సరానికి ₹6,000 నేరుగా ఆదాయం సహాయం.",

"pmfby_title": "ప్రధాన మంత్రి పంట బీమా యోజన",
"pmfby_desc": "ప్రకృతి విపత్తుల వల్ల పంట నష్టానికి బీమా రక్షణ.",

"soilhealth_title": "మట్టి ఆరోగ్య కార్డ్ పథకం",
"soilhealth_desc": "మట్టి పరీక్ష మరియు సారాంశ విశ్లేషణ.",

"weather_dashboard": "వాతావరణ డాష్‌బోర్డ్",
"enter_place": "నగరం / స్థలం నమోదు చేయండి",
"add_place": "స్థలం చేర్చండి",
"my_location": "నా లొకేషన్",
"latest_weather_news": "తాజా వాతావరణ వార్తలు",
"temp": "ఉష్ణోగ్రత",
"humidity": "ఆర్ద్రత",
"wind": "గాలి",
"rain": "వర్షం",
"condition": "స్థితి",
"place_not_found": "స్థలం కనబడలేదు",
"enter_valid_place": "సరైన స్థలం నమోదు చేయండి",

"soil": "మట్టి సమాచారం",
"soil_title": "మట్టి సమాచార వ్యవస్థ",
"select_soil": "-- మట్టి రకం ఎంచుకోండి --",
"loading_soil": "మట్టి సమాచారం లోడ్ అవుతోంది...",
"advantages": "ప్రయోజనాలు",
"limitations": "పరిమితులు",
"best_crops": "ఉత్తమ పంటలు",
"fertilizer_tips": "ఎరువు సూచనలు",
"pest_management": "పురుగు నియంత్రణ",
"error_loading_soil": "మట్టి డేటా లోడ్ కాలేదు"
    },

    "kn": {
        "login": "ಲಾಗಿನ್",
        "register": "ನೋಂದಣಿ",
        "email": "ಇಮೇಲ್",
        "phone": "ಫೋನ್",
        "name": "ಹೆಸರು",
        "state": "ರಾಜ್ಯ",
        "dashboard": "ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
        "subtitle": "ನಿಮ್ಮ ಸ್ಮಾರ್ಟ್ ಕೃಷಿ ಸಹಾಯಕ",

        "disease": "ರೋಗ ಪತ್ತೆ",
        "disease_desc": "ಬೆಳೆ ರೋಗಗಳನ್ನು ಗುರುತಿಸಿ.",
        "disease_btn": "ಸ್ಕ್ಯಾನ್",

        "schemes": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        "schemes_desc": "ಕೇಂದ್ರ & ರಾಜ್ಯ ಯೋಜನೆಗಳು.",
        "schemes_btn": "ನೋಡಿ",

        "market": "ಹತ್ತಿರದ ಮಾರುಕಟ್ಟೆ",
        "market_desc": "ಮಾರಾಟಕ್ಕೆ ಮಾರುಕಟ್ಟೆಗಳು.",
        "market_btn": "ಮಾರುಕಟ್ಟೆ",

        "prices": "ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು",
        "prices_desc": "ಇಂದಿನ ಬೆಲೆಗಳು.",
        "prices_btn": "ಬೆಲೆ",

        "search_crop": "ಬೆಳೆ ಹುಡುಕಿ...",
        "crop_not_found": "❌ ಬೆಳೆ ಸಿಗಲಿಲ್ಲ",
        "price": "ಬೆಲೆ",

        "weather": "ಹವಾಮಾನ",
        "weather_desc": "ನೇರ ಹವಾಮಾನ ಮಾಹಿತಿ.",
        "weather_btn": "ಲೊಕೇಶನ್",

        "soil": "ಮಣ್ಣು ಮಾಹಿತಿ",
        "soil_desc": "ಮಣ್ಣು ಗುಣಗಳು & ಬೆಳೆಗಳು.",
        "soil_btn": "ಮಣ್ಣು ನೋಡಿ",

        "calendar": "ರೈತ ಕ್ಯಾಲೆಂಡರ್",
"select_state": "-- ರಾಜ್ಯವನ್ನು ಆಯ್ಕೆಮಾಡಿ --",
"loading_calendar": "ಸೀಸನ್ ಯೋಜನೆ ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
"sowing": "ಬಿತ್ತನೆ",
"planting": "ನೆಡುವುದು",
"harvest": "ಕೊಯ್ಲು",
"error_loading_calendar": "ಕ್ಯಾಲೆಂಡರ್ ಲೋಡ್ ಮಾಡಲು ಸಾಧ್ಯವಾಗಲಿಲ್ಲ",
"calendar_btn": "ಕ್ಯಾಲೆಂಡರ್ ತೆರೆಯಿರಿ",
        "ai": "ನನ್ನ ರೈತ AI",
        "ai_desc": "ಕೃಷಿ ಮಾರ್ಗದರ್ಶಕ AI.",
        "ai_btn": "ಚಾಟ್",

        "schemes": "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
"central_schemes": "ಕೇಂದ್ರ ಸರ್ಕಾರದ ಯೋಜನೆಗಳು",
"select_state": "ನಿಮ್ಮ ರಾಜ್ಯವನ್ನು ಆಯ್ಕೆಮಾಡಿ",
"official_site": "ಅಧಿಕೃತ ವೆಬ್‌ಸೈಟ್",

"pmkisan_title": "ಪ್ರಧಾನಮಂತ್ರಿ ಕಿಸಾನ್ ಸಮ್ಮಾನ್ ನಿಧಿ (PM-KISAN)",
"pmkisan_desc": "ರೈತರಿಗೆ ಪ್ರತಿವರ್ಷ ₹6,000 ನೇರ ಆದಾಯ ನೆರವು.",

"pmfby_title": "ಪ್ರಧಾನಮಂತ್ರಿ ಬೆಳೆ ವಿಮಾ ಯೋಜನೆ",
"pmfby_desc": "ಪ್ರಕೃತಿ ವಿಪತ್ತುಗಳಿಂದ ಬೆಳೆ ನಷ್ಟಕ್ಕೆ ವಿಮಾ ರಕ್ಷಣೆ.",

"soilhealth_title": "ಮಣ್ಣಿನ ಆರೋಗ್ಯ ಕಾರ್ಡ್ ಯೋಜನೆ",
"soilhealth_desc": "ಮಣ್ಣಿನ ಪರೀಕ್ಷೆ ಮತ್ತು ಉರ್ವರತೆ ಮೌಲ್ಯಮಾಪನ.",

"weather_dashboard": "ಹವಾಮಾನ ಡ್ಯಾಶ್‌ಬೋರ್ಡ್",
"enter_place": "ನಗರ / ಸ್ಥಳ ನಮೂದಿಸಿ",
"add_place": "ಸ್ಥಳ ಸೇರಿಸಿ",
"my_location": "ನನ್ನ ಸ್ಥಳ",
"latest_weather_news": "ಇತ್ತೀಚಿನ ಹವಾಮಾನ ಸುದ್ದಿ",
"temp": "ತಾಪಮಾನ",
"humidity": "ಆದ್ರತೆ",
"wind": "ಗಾಳಿ",
"rain": "ಮಳೆ",
"condition": "ಸ್ಥಿತಿ",
"place_not_found": "ಸ್ಥಳ ಸಿಗಲಿಲ್ಲ",
"enter_valid_place": "ಸರಿಯಾದ ಸ್ಥಳ ನಮೂದಿಸಿ",

"soil": "ಮಣ್ಣು ಮಾಹಿತಿ",
"soil_title": "ಮಣ್ಣು ಮಾಹಿತಿ ವ್ಯವಸ್ಥೆ",
"select_soil": "-- ಮಣ್ಣಿನ ಪ್ರಕಾರ ಆಯ್ಕೆಮಾಡಿ --",
"loading_soil": "ಮಣ್ಣು ಮಾಹಿತಿ ಲೋಡ್ ಆಗುತ್ತಿದೆ...",
"advantages": "ಪ್ರಯೋಜನಗಳು",
"limitations": "ಮಿತಿಗಳು",
"best_crops": "ಉತ್ತಮ ಬೆಳೆಗಳು",
"fertilizer_tips": "ಸಾರ ಸಲಹೆಗಳು",
"pest_management": "ಕೀಟ ನಿಯಂತ್ರಣ",
"error_loading_soil": "ಮಣ್ಣು ಡೇಟಾ ಲೋಡ್ ಆಗಲಿಲ್ಲ"
    },

    "hi": {
        "login": "लॉगिन",
        "register": "पंजीकरण",
        "email": "ईमेल",
        "phone": "फ़ोन",
        "name": "नाम",
        "state": "राज्य",
        "dashboard": "डैशबोर्ड",
        "subtitle": "आपका स्मार्ट कृषि साथी",

        "disease": "रोग पहचान",
        "disease_desc": "फसल रोग पहचानें.",
        "disease_btn": "स्कैन",

        "schemes": "सरकारी योजनाएँ",
        "schemes_desc": "केंद्र & राज्य योजनाएँ.",
        "schemes_btn": "देखें",

        "market": "नज़दीकी बाज़ार",
        "market_desc": "फसल बेचने के बाज़ार.",
        "market_btn": "बाज़ार",

        "prices": "बाज़ार मूल्य",
        "prices_desc": "आज के मूल्य.",
        "prices_btn": "मूल्य",

        "search_crop": "फसल खोजें...",
        "crop_not_found": "❌ फसल नहीं मिली",
        "price": "मूल्य",

        "weather": "मौसम",
        "weather_desc": "लाइव मौसम जानकारी.",
        "weather_btn": "लोकेशन",

        "soil": "मिट्टी जानकारी",
        "soil_desc": "मिट्टी लाभ & उपयुक्त फसलें.",
        "soil_btn": "मिट्टी देखें",

        "calendar": "किसान कैलेंडर",
"select_state": "-- राज्य चुनें --",
"loading_calendar": "सीज़न योजना लोड हो रही है...",
"sowing": "बुवाई",
"planting": "रोपण",
"harvest": "कटाई",
"error_loading_calendar": "कैलेंडर लोड नहीं हो सका",
"calendar_btn": "कैलेंडर खोलें",

        "ai": "मेरा किसान AI",
        "ai_desc": "कृषि मार्गदर्शक AI.",
        "ai_btn": "चैट",

        "schemes": "सरकारी योजनाएँ",
"central_schemes": "केंद्रीय सरकारी योजनाएँ",
"select_state": "अपने राज्य का चयन करें",
"official_site": "आधिकारिक वेबसाइट",

"pmkisan_title": "प्रधान मंत्री किसान सम्मान निधि (PM-KISAN)",
"pmkisan_desc": "किसानों के लिए प्रति वर्ष ₹6,000 की आय सहायता।",

"pmfby_title": "प्रधान मंत्री फसल बीमा योजना",
"pmfby_desc": "प्राकृतिक आपदाओं से फसल नुकसान पर बीमा सुरक्षा।",

"soilhealth_title": "मृदा स्वास्थ्य कार्ड योजना",
"soilhealth_desc": "मृदा परीक्षण और उर्वरता मूल्यांकन।",

"weather_dashboard": "मौसम डैशबोर्ड",
"enter_place": "शहर / स्थान दर्ज करें",
"add_place": "स्थान जोड़ें",
"my_location": "मेरी लोकेशन",
"latest_weather_news": "ताज़ा मौसम समाचार",
"temp": "तापमान",
"humidity": "नमी",
"wind": "हवा",
"rain": "बारिश",
"condition": "स्थिति",
"place_not_found": "स्थान नहीं मिला",
"enter_valid_place": "कृपया सही स्थान दर्ज करें",

"soil": "मिट्टी जानकारी",
"soil_title": "मिट्टी जानकारी प्रणाली",
"select_soil": "-- मिट्टी का प्रकार चुनें --",
"loading_soil": "मिट्टी जानकारी लोड हो रही है...",
"advantages": "लाभ",
"limitations": "सीमाएँ",
"best_crops": "उपयुक्त फसलें",
"fertilizer_tips": "उर्वरक सुझाव",
"pest_management": "कीट नियंत्रण",
"error_loading_soil": "मिट्टी डेटा लोड नहीं हुआ"
    }
}
        

# ---------------- Flask App ----------------

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ✅ Inject language globally
@app.context_processor
def inject_globals():

    lang = session.get("lang", "en")

    def translate(text):
        if not text:
            return text
        if lang == "en":
            return text
        try:
            return translate_text(text, lang)
        except:
            return text

    return dict(
        text=LANGUAGES.get(lang, LANGUAGES["en"]),
        translate=translate,
        current_lang=lang
    )

# ---------------- Chatbot Logic ----------------

def chatbot_reply(user_text, lang="en"):
    try:
        user_lower = user_text.lower().strip()

        greetings = ["hi", "hello", "hey", "hii", "good morning", "good evening"]
        if user_lower in greetings:
            reply = "Hello 👋 I’m your Farmer AI assistant. Ask me about crops, fertilizers, diseases, irrigation, or pests 🌾"
            if lang != "en":
                reply = translator.translate(reply, dest=lang).text
            return reply

        thanks = ["thank you", "thanks", "thankyou", "thx"]
        if user_lower in thanks:
            reply = "You’re welcome 😊 Happy to help with your farming questions!"
            if lang != "en":
                reply = translator.translate(reply, dest=lang).text
            return reply

        if lang != "en":
            user_text_en = translator.translate(user_text, dest="en").text
        else:
            user_text_en = user_text

        user_embedding = chat_model.encode([user_text_en])
        scores = cosine_similarity(user_embedding, chat_embeddings)[0]
        best_idx = scores.argmax()

        answer_en = chat_answers[best_idx]

        if lang != "en":
            final_answer = translator.translate(answer_en, dest=lang).text
        else:
            final_answer = answer_en

        return final_answer

    except Exception as e:
        print("🔥 CHATBOT ERROR:", e)
        return "Sorry, I couldn't process your question."

# ---------------- Database ----------------

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///farmers.db'
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

class Farmer(db.Model):
    __tablename__ = "farmer"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    state = db.Column(db.String(100))
    email = db.Column(db.String(100), unique=True, nullable=False)
    phone = db.Column(db.String(20), nullable=False)

with app.app_context():
    db.create_all()

# ---------------- Market Prices ----------------

df_prices = pd.read_csv("data/market_prices.csv")

# ---------------- Weather API ----------------

API_KEY = "546fa3458a4cb31dd7b7b47cb274188b"

# ---------------- Routes ----------------
#-----------------Welcome-----------------#
@app.route("/")
def welcome():
    return render_template("welcome.html")

@app.route("/set-language", methods=["POST"])
def set_language():
    lang = request.form.get("lang")
    session["lang"] = lang
    return redirect(url_for("login"))
#------------------------Registration-------------------------#
@app.route("/register", methods=['GET','POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        state = request.form['state']
        email = request.form['email']
        phone = request.form['phone']

        existing_user = Farmer.query.filter_by(email=email).first()
        if existing_user:
            lang = session.get("lang", "en")
            flash(translator.translate("Email already registered!", dest=lang).text)
            return redirect(url_for('login'))

        farmer = Farmer(name=name, state=state, email=email, phone=phone)
        db.session.add(farmer)
        db.session.commit()

        session['user_email'] = email
        return redirect(url_for('dashboard'))

    return render_template("register.html")
#--------------------Login-------------------#
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email'].strip()
        phone = request.form['phone'].strip()

        user = Farmer.query.filter_by(email=email, phone=phone).first()

        if user:
            session['user_email'] = email
            return redirect(url_for('dashboard'))
        else:
            lang = session.get("lang", "en")
            flash(translator.translate("Invalid email or phone!", dest=lang).text)
            return redirect(url_for('login'))

    return render_template('login.html')
#------------------Logout-------------------#
@app.route('/logout')
def logout():
    session.pop('user_email', None)
    return redirect(url_for('welcome'))
#----------------Dashboard-----------------#
@app.route("/dashboard")
def dashboard():
    if 'user_email' not in session:
        lang = session.get("lang", "en")
        flash(translator.translate("Please login first!", dest=lang).text)
        return redirect(url_for('login'))

    user = Farmer.query.filter_by(email=session['user_email']).first()
    return render_template("dashboard.html", user=user)

# ---------------- Disease Detector ----------------

@app.route("/disease", methods=["GET", "POST"])
def disease():
    if request.method == "POST":
        if "image" not in request.files:
            return render_template("disease.html", error="No image uploaded")

        img = request.files["image"]
        if img.filename == "":
            return render_template("disease.html", error="No image selected")

        UPLOAD_FOLDER = "static/uploads"
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        filename = secure_filename(img.filename)
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img.save(img_path)

        disease_name, confidence = predict_disease(img_path)

        return render_template("disease.html",
                               disease=disease_name,
                               confidence=confidence,
                               image=f"uploads/{filename}")

    return render_template("disease.html")

# ---------------- Market ----------------

@app.route("/market")
def market():
    markets = [
        {"market": "Koyambedu Market", "district": "Chennai", "state": "Tamil Nadu"},
        {"market": "Gandhi Market", "district": "Trichy", "state": "Tamil Nadu"},
        {"market": "Uzhavar Sandhai", "district": "Madurai", "state": "Tamil Nadu"},
        {"market": "KR Market", "district": "Bangalore", "state": "Karnataka"},
        {"market": "Madiwala Market", "district": "Bangalore", "state": "Karnataka"},
        {"market": "Mahatma Phule Market", "district": "Mumbai", "state": "Maharashtra"},
        {"market": "Bowenpally Market", "district": "Hyderabad", "state": "Telangana"},
        {"market": "Vashi Market", "district": "Navi Mumbai", "state": "Maharashtra"}
    ]

    return render_template("market.html", markets=markets)
#---------------prices---------------------

@app.route("/prices")
def prices():
    return render_template("prices.html")
@app.route("/get-price")
def get_price():
    crop = request.args.get("crop", "").strip()
    lang = session.get("lang", "en")

    if not crop:
        return jsonify({"error": LANGUAGES[lang]["crop_not_found"]})

    # Filter matching crop
    result = df_prices[
        df_prices["commodity"].str.contains(crop, case=False, na=False)
    ]

    if result.empty:
        return jsonify({"error": LANGUAGES[lang]["crop_not_found"]})

    # ✅ Get latest record safely
    latest = result.iloc[-1]

    crop_name = latest["commodity"]
    market_name = latest["market"]
    district = latest["district"]
    price = latest["modal_price"]

    # ✅ Translate if needed
    if lang != "en":
        try:
            crop_name = translator.translate(crop_name, dest=lang).text
            market_name = translator.translate(market_name, dest=lang).text
            district = translator.translate(district, dest=lang).text
        except:
            pass

    return jsonify({
        "crop": crop_name,
        "price": price,
        "market": market_name,
        "district": district
    })


        
           


    

# ---------------- Schemes ----------------
@app.route("/schemes")
def schemes():
    lang = session.get("lang", "en")

    schemes_data = {
        "Kerala": ["Karshaka Welfare Scheme"],
        "Tamil Nadu": ["Uzhavar Santhai"],
        "Karnataka": ["Raitha Siri"],
        "Andhra Pradesh": ["YSR Rythu Bharosa"],
        "Telangana": ["Rythu Bandhu"]
    }

    # ✅ Translate dynamically
    if lang != "en":
        translated = {}
        for state, scheme_list in schemes_data.items():
            translated_state = translate_text(state, lang)
            translated[translated_state] = [
                translate_text(s, lang) for s in scheme_list
            ]
        schemes_data = translated

    return render_template("schemes.html", schemes_json=schemes_data)


# ---------------- Weather ----------------

@app.route("/weather")
def weather():
    return render_template("weather.html")

@app.route("/get-weather")
def get_weather_api():
    city = request.args.get("city")

    if not city:
        return jsonify({"error": "City not provided"})

    try:
        current_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        current = requests.get(current_url).json()

        return jsonify({
            "city": city,
            "temp": current["main"]["temp"],
            "humidity": current["main"]["humidity"],
            "condition": current["weather"][0]["description"]
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ---------------- Farmer AI ----------------

@app.route("/farmer-ai")
def farmer_ai():
    return render_template("farmer_ai.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_msg = data.get("message", "")
    lang = data.get("lang", "en")

    reply = chatbot_reply(user_msg, lang)

    return jsonify({
        "status": "success",
        "reply": reply
    })

# ---------------- Soil ----------------

@app.route("/soil")
def soil():
    return render_template("soil.html")

@app.route("/get-soil")
def get_soil():
    soil_type = request.args.get("soil")

    df_soil = pd.read_csv("data/soil_data.csv")
    result = df_soil[df_soil["soil_type"] == soil_type]

    if result.empty:
        return jsonify({"error": "Soil data not found"})

    row = result.iloc[0]

    return jsonify({
        "advantages": row["advantages"],
        "limitations": row["limitations"],
        "best_crops": row["best_crops"],
        "fertilizer_tips": row["fertilizer_tips"],
        "pest_management": row["pest_management"]
    })

# ---------------- Calendar ----------------

@app.route("/calendar")
def calendar():
    return render_template("calendar.html")

@app.route("/get-calendar")
def get_calendar():
    state = request.args.get("state")

    df_cal = pd.read_csv("data/farming_calendar.csv")
    result = df_cal[df_cal["state"] == state]

    if result.empty:
        return jsonify({"error": "No calendar data found"})

    return jsonify(result.to_dict(orient="records"))
#-------------------translator-----------------
@app.route("/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_msg = data.get("message", "")

    # ✅ NOW uses session language
    lang = session.get("lang", "en")

    reply = chatbot_reply(user_msg, lang)

    return jsonify({
        "status": "success",
        "reply": reply
    })

    
# ---------------- Run ----------------

if __name__ == "__main__":
    app.run(debug=True)  