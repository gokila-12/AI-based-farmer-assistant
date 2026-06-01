function changeLanguage(lang) {
    const greetings = {
        en: "Hey Farmer!!!",
        ml: "ഹേ കർഷകാ!!!",
        ta: "வணக்கம் விவசாயி!!!",
        te: "హే రైతు!!!",
        kn: "ಹೇ ರೈತ!!!",
        hi: "हे किसान!!!"
    };
    document.getElementById("greeting").innerText = greetings[lang];
}
function changeDashboardLang() {

    const lang = document.getElementById("language").value;

    const text = {
        en: {
            title: "Smart Farmer Assistant",
            disease: "Disease Detector",
            market: "Near By Market",
            schemes: "Government Schemes",
            prices: "Market Prices",
            weather: "Weather Forecast",
            farmerai: "My Farmer AI"
        },

        ml: {
            title: "സ്മാർട്ട് ഫാർമർ അസിസ്റ്റന്റ്",
            disease: "രോഗം കണ്ടെത്തൽ",
            market: "അടുത്തുള്ള മാർക്കറ്റ്",
            schemes: "സർക്കാർ പദ്ധതികൾ",
            prices: "വിപണി വിലകൾ",
            weather: "കാലാവസ്ഥ പ്രവചനം",
            farmerai: "എന്റെ ഫാർമർ AI"
        },

        ta: {
            title: "ஸ்மார்ட் விவசாய உதவியாளர்",
            disease: "நோய் கண்டறிதல்",
            market: "அருகிலுள்ள சந்தை",
            schemes: "அரசு திட்டங்கள்",
            prices: "சந்தை விலைகள்",
            weather: "வானிலை முன்னறிவிப்பு",
            farmerai: "என் விவசாய AI"
        },

        te: {
            title: "స్మార్ట్ రైతు సహాయకుడు",
            disease: "రోగ నిర్ధారణ",
            market: "సమీప మార్కెట్",
            schemes: "ప్రభుత్వ పథకాలు",
            prices: "మార్కెట్ ధరలు",
            weather: "వాతావరణ సూచన",
            farmerai: "నా రైతు AI"
        },

        kn: {
            title: "ಸ್ಮಾರ್ಟ್ ರೈತ ಸಹಾಯಕ",
            disease: "ರೋಗ ಪತ್ತೆ",
            market: "ಹತ್ತಿರದ ಮಾರುಕಟ್ಟೆ",
            schemes: "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
            prices: "ಮಾರುಕಟ್ಟೆ ಬೆಲೆಗಳು",
            weather: "ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ",
            farmerai: "ನನ್ನ ರೈತ AI"
        },

        hi: {
            title: "स्मार्ट किसान सहायक",
            disease: "रोग पहचान",
            market: "नज़दीकी बाज़ार",
            schemes: "सरकारी योजनाएँ",
            prices: "बाज़ार मूल्य",
            weather: "मौसम पूर्वानुमान",
            farmerai: "मेरा किसान AI"
        }
    };

    document.getElementById("title").innerText = text[lang].title;
    document.getElementById("disease").innerText = "🌿 " + text[lang].disease;
    document.getElementById("market").innerText = "🏪 " + text[lang].market;
    document.getElementById("schemes").innerText = "📢 " + text[lang].schemes;
    document.getElementById("prices").innerText = "💰 " + text[lang].prices;
    document.getElementById("weather").innerText = "🌦 " + text[lang].weather;
    document.getElementById("farmerai").innerText = "🤖 " + text[lang].farmerai;
}
document.addEventListener("DOMContentLoaded", function() {
    changeDashboardLang();
});

function showMarkets(pos){
    marketList.innerHTML="";

    let userMarkets=JSON.parse(localStorage.getItem("userMarkets")) || [];
    let combined=[...allMarkets, ...userMarkets];

    combined
    .map(m=>{
        const d=getDistance(pos.coords.latitude,pos.coords.longitude,m.lat,m.lon);
        return {...m, distance:d};
    })
    .filter(m=>m.distance<=20) // 20 km radius
    .sort((a,b)=>a.distance-b.distance)
    .forEach(m=>{
        const time=Math.round((m.distance/30)*60);
        const div=document.createElement("div");
        div.className="market-card";
        div.innerHTML=`
            <h4>${m.name}</h4>
            <p>${text[currentLang].distance}: ${m.distance.toFixed(2)} km</p>
            <p>${text[currentLang].time}: ${time} mins</p>
        `;
        marketList.appendChild(div);
    });
}
