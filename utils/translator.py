from googletrans import Translator

translator = Translator()

def translate_text(text, lang="en"):
    try:
        if lang == "en":
            return text

        return translator.translate(text, dest=lang).text

    except Exception as e:
        print("🔥 TRANSLATION ERROR:", e)
        return text  # fallback (no crash)
    
    