async function sendMessage() {
    const input = document.getElementById("user-input");
    const message = input.value.trim();

    if (!message) return;

    addMessage(message, "user");
    input.value = "";

    addMessage("🤖 AI is thinking...", "bot");

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();

        removeThinking();

        if (data.reply) {
            addMessage(data.reply, "bot");
        } else {
            addMessage("⚠️ AI returned an empty response.", "bot");
        }

    } catch (error) {
        removeThinking();
        addMessage("⚠️ Error connecting to AI server.", "bot");
        console.error("Chat Error:", error);
    }
}


function addMessage(text, sender) {
    const chatBox = document.getElementById("chat-box");

    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${sender}`;
    msgDiv.innerText = text;

    chatBox.appendChild(msgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}


function removeThinking() {
    const chatBox = document.getElementById("chat-box");
    const messages = chatBox.getElementsByClassName("bot");

    if (messages.length > 0) {
        const lastMsg = messages[messages.length - 1];
        if (lastMsg.innerText.includes("thinking")) {
            lastMsg.remove();
        }
    }
}
