/**
 * Xwave — AI Voice Agent Frontend
 * White + Butter Yellow Theme · Aether-inspired layout
 */

const API_BASE = "http://localhost:8000";

// ── State ────────────────────────────────────────────────────────────────────
let sessionId = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// ── DOM ──────────────────────────────────────────────────────────────────────
const micBtn        = document.getElementById("mic-btn");
const micLabel      = document.getElementById("mic-label");
const recIndicator  = document.getElementById("rec-indicator");
const chatArea      = document.getElementById("chat-area");
const transcriptText= document.getElementById("transcript-text");
const sentimentChip = document.getElementById("sentiment-chip");
const sentimentEmoji= document.getElementById("sentiment-emoji");
const sentimentLabel= document.getElementById("sentiment-label");
const sentimentScore= document.getElementById("sentiment-score");
const escalationBanner = document.getElementById("escalation-banner");
const suggestions   = document.getElementById("suggestions");
const debugToggle   = document.getElementById("debug-toggle");
const debugContent  = document.getElementById("debug-content");
const debugOutput   = document.getElementById("debug-output");

// ── Floating Particles ───────────────────────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById("particles");
  const ctx = canvas.getContext("2d");
  let particles = [];

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener("resize", resize);

  for (let i = 0; i < 45; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 2 + 0.5,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      opacity: Math.random() * 0.25 + 0.05,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(245, 197, 24, ${p.opacity})`;
      ctx.fill();
      p.x += p.dx;
      p.y += p.dy;
      if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── Session Reset (available programmatically) ──────────────────────────────
function resetSession() {
  sessionId = null;
  chatArea.innerHTML = "";
  transcriptText.textContent = "";
  sentimentChip.classList.add("hidden");
  escalationBanner.classList.add("hidden");
  document.body.classList.remove("has-messages");
  sentimentChip.className = "sentiment-chip hidden";
}

// ── Debug Toggle ─────────────────────────────────────────────────────────────
debugToggle.addEventListener("click", () => {
  debugContent.classList.toggle("hidden");
});

// ── Quick Suggestion Chips (disabled — no chips in current UI) ──────────────

// ── Mic Button — Hold to Record ──────────────────────────────────────────────
micBtn.addEventListener("mousedown",  startRecording);
micBtn.addEventListener("touchstart", (e) => { e.preventDefault(); startRecording(); });
micBtn.addEventListener("mouseup",    stopAndSend);
micBtn.addEventListener("mouseleave", stopAndSend);
micBtn.addEventListener("touchend",   stopAndSend);

async function startRecording() {
  if (isRecording) return;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    audioChunks = [];
    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };
    mediaRecorder.start(100);
    isRecording = true;
    micBtn.classList.add("recording");
    micLabel.textContent = "RELEASE";
    recIndicator.classList.remove("hidden");
  } catch (err) {
    addMessage("system", "Mic access denied. Please allow microphone access in your browser settings.");
  }
}

async function stopAndSend() {
  if (!isRecording || !mediaRecorder) return;
  isRecording = false;
  micBtn.classList.remove("recording");
  micLabel.textContent = "PROCESSING";
  micBtn.disabled = true;
  recIndicator.classList.add("hidden");

  await new Promise((resolve) => {
    mediaRecorder.onstop = resolve;
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach((t) => t.stop());
  });

  if (audioChunks.length === 0) { resetMic(); return; }
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  await processVoiceTurn(blob);
  resetMic();
}

function resetMic() {
  micLabel.textContent = "HOLD TO SPEAK";
  micBtn.disabled = false;
}

// ── Process Text Turn (from suggestion chips) ────────────────────────────────
async function processTextTurn(text) {
  document.body.classList.add("has-messages");
  addMessage("user", text);
  const thinkingId = addThinking();
  transcriptText.textContent = `You said: "${text}"`;

  try {
    const body = { message: text };
    if (sessionId) body.session_id = sessionId;

    const res = await fetch(`${API_BASE}/agent/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();

    sessionId = data.session_id;

    updateSentiment(data.sentiment);
    removeThinking(thinkingId);

    const isEscalation = data.action === "escalate";
    addMessage(isEscalation ? "escalated" : "agent", data.agent_response);
    if (isEscalation) escalationBanner.classList.remove("hidden");

    // Speak the response
    try {
      const voiceKey = data.sentiment?.score < -0.4 ? "empathetic" : "default";
      const audioRes = await fetch(`${API_BASE}/voice/speak?text=${encodeURIComponent(data.agent_response)}&voice=${voiceKey}`);
      const audioBlob = await audioRes.blob();
      new Audio(URL.createObjectURL(audioBlob)).play().catch(() => {});
    } catch {}

    updateDebug(data);
  } catch (err) {
    removeThinking(thinkingId);
    addMessage("system", `Error: ${err.message}`);
  }
}

// ── Process Voice Turn ───────────────────────────────────────────────────────
async function processVoiceTurn(audioBlob) {
  document.body.classList.add("has-messages");
  transcriptText.textContent = "Transcribing...";
  const thinkingId = addThinking();

  try {
    const formData = new FormData();
    formData.append("audio", audioBlob, "recording.webm");
    if (sessionId) formData.append("session_id", sessionId);

    const res = await fetch(`${API_BASE}/agent/chat/voice`, {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error(`API ${res.status}`);

    const h = res.headers;
    const newSession = h.get("X-Session-Id");
    const turn       = h.get("X-Turn-Number");
    const sentLabel  = h.get("X-Sentiment");
    const sentScore  = parseFloat(h.get("X-Sentiment-Score") || "0");
    const action     = h.get("X-Action");
    const transcript = h.get("X-Transcript");

    if (newSession) {
      sessionId = newSession;
    }

    transcriptText.textContent = transcript ? `You said: "${transcript}"` : "";
    removeThinking(thinkingId);

    if (transcript) addMessage("user", transcript);
    updateSentiment({ label: sentLabel, score: sentScore });

    // Get agent text from conversation endpoint
    let agentText = "...";
    try {
      const convRes = await fetch(`${API_BASE}/agent/conversation/${sessionId}`);
      const conv = await convRes.json();
      agentText = conv.turns?.[conv.turns.length - 1]?.agent || agentText;
    } catch {}

    const isEscalation = action === "escalate";
    addMessage(isEscalation ? "escalated" : "agent", agentText);
    if (isEscalation) escalationBanner.classList.remove("hidden");

    // Play audio response
    const audioBuffer = await res.arrayBuffer();
    const blob2 = new Blob([audioBuffer], { type: "audio/mpeg" });
    new Audio(URL.createObjectURL(blob2)).play().catch(() => {});

    updateDebug({
      session_id: sessionId, turn, action,
      sentiment: { label: sentLabel, score: sentScore },
    });

  } catch (err) {
    removeThinking(thinkingId);
    addMessage("system", `Error: ${err.message}`);
    transcriptText.textContent = "";
  }
}

// ── UI Helpers ───────────────────────────────────────────────────────────────
function addMessage(type, text) {
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = text;
  chatArea.appendChild(div);
  const main = document.querySelector(".main");
  main.scrollTop = main.scrollHeight;
}

let thinkingCounter = 0;
function addThinking() {
  const id = `thinking-${thinkingCounter++}`;
  const div = document.createElement("div");
  div.className = "msg thinking";
  div.id = id;
  div.innerHTML = '<div class="dot-loader"><span></span><span></span><span></span></div>';
  chatArea.appendChild(div);
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function updateSentiment(sentiment) {
  if (!sentiment) return;
  const { label, score } = sentiment;

  sentimentChip.classList.remove("hidden", "positive", "negative");
  sentimentEmoji.textContent = getEmoji(label, score);
  sentimentLabel.textContent = label || "NEUTRAL";
  sentimentScore.textContent = typeof score === "number"
    ? (score > 0 ? "+" : "") + score.toFixed(2)
    : "—";

  if (score > 0.2) sentimentChip.classList.add("positive");
  else if (score < -0.2) sentimentChip.classList.add("negative");
}

function getEmoji(label, score) {
  const map = {
    HAPPY: "😄", POSITIVE: "😊", CALM: "🙂",
    NEUTRAL: "😐", CONCERNED: "😕", FRUSTRATED: "😟", UPSET: "😤",
  };
  return map[label] || "😐";
}

function updateDebug(data) {
  debugOutput.textContent = JSON.stringify({
    ...data,
    langsmith: "smith.langchain.com → voice-agent project",
  }, null, 2);
}
