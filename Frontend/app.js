/**
 * Xwave v2 — AI Voice Agent Frontend
 * Features: Text + Voice input, RAG source display, Tool action feed,
 *           Live sentiment strip, Debug panel, Particles
 */

const API_BASE = "http://localhost:8000";

// ── State ────────────────────────────────────────────────────────────────────
let sessionId = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// ── DOM ──────────────────────────────────────────────────────────────────────
const micBtn         = document.getElementById("mic-btn");
const micLabel       = document.getElementById("mic-label");
const chatArea       = document.getElementById("chat-area");
const transcriptText = document.getElementById("transcript-text");
const textInput      = document.getElementById("text-input");
const sendBtn        = document.getElementById("send-btn");
const statusDot      = document.getElementById("status-dot");
const statusText     = document.getElementById("status-text");
const newSessionBtn  = document.getElementById("new-session-btn");

// Sentiment
const sentimentStrip = document.getElementById("sentiment-strip");
const sentimentFill  = document.getElementById("sentiment-fill");
const sentimentThumb = document.getElementById("sentiment-thumb");
const sentimentEmoji = document.getElementById("sentiment-emoji");
const sentimentLabel = document.getElementById("sentiment-label");
const sentimentScore = document.getElementById("sentiment-score");

const escalationBanner = document.getElementById("escalation-banner");
const debugToggle    = document.getElementById("debug-toggle");
const debugPanel     = document.getElementById("debug-panel");
const debugOutput    = document.getElementById("debug-output");

// ── Particles ────────────────────────────────────────────────────────────────
(function initParticles() {
  const canvas = document.getElementById("particles");
  const ctx = canvas.getContext("2d");
  let particles = [];

  function resize() { canvas.width = window.innerWidth; canvas.height = window.innerHeight; }
  resize();
  window.addEventListener("resize", resize);

  for (let i = 0; i < 40; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      r: Math.random() * 1.8 + 0.4,
      dx: (Math.random() - 0.5) * 0.25,
      dy: (Math.random() - 0.5) * 0.25,
      opacity: Math.random() * 0.2 + 0.04,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(245, 197, 24, ${p.opacity})`;
      ctx.fill();
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0 || p.x > canvas.width) p.dx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.dy *= -1;
    });
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── Session ──────────────────────────────────────────────────────────────────
newSessionBtn.addEventListener("click", () => {
  sessionId = null;
  chatArea.innerHTML = "";
  transcriptText.textContent = "";
  sentimentStrip.classList.add("hidden");
  escalationBanner.classList.add("hidden");
  document.body.classList.remove("has-messages");
  setStatus("ready", "Ready");
});

// ── Debug Toggle ─────────────────────────────────────────────────────────────
debugToggle.addEventListener("click", () => debugPanel.classList.toggle("hidden"));

// ── Text Input ───────────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendText);
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendText(); }
});

function sendText() {
  const text = textInput.value.trim();
  if (!text) return;
  textInput.value = "";
  processTextTurn(text);
}

// ── Mic — Hold to Record ─────────────────────────────────────────────────────
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
    mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) audioChunks.push(e.data); };
    mediaRecorder.start(100);
    isRecording = true;
    micBtn.classList.add("recording");
    micLabel.textContent = "...";
    setStatus("processing", "Listening...");
  } catch (err) {
    addMessage("system", "Mic access denied. Check browser permissions.");
  }
}

async function stopAndSend() {
  if (!isRecording || !mediaRecorder) return;
  isRecording = false;
  micBtn.classList.remove("recording");
  micLabel.textContent = "WAIT";
  micBtn.disabled = true;

  await new Promise(resolve => {
    mediaRecorder.onstop = resolve;
    mediaRecorder.stop();
    mediaRecorder.stream.getTracks().forEach(t => t.stop());
  });

  if (audioChunks.length === 0) { resetMic(); return; }
  const blob = new Blob(audioChunks, { type: "audio/webm" });
  await processVoiceTurn(blob);
  resetMic();
}

function resetMic() {
  micLabel.textContent = "HOLD";
  micBtn.disabled = false;
}

// ── Status Helper ────────────────────────────────────────────────────────────
function setStatus(state, text) {
  statusText.textContent = text;
  statusDot.className = "status-dot";
  if (state === "processing") statusDot.classList.add("processing");
  else if (state === "error") statusDot.classList.add("error");
}

// ── Process Text Turn ────────────────────────────────────────────────────────
async function processTextTurn(text) {
  document.body.classList.add("has-messages");
  addMessage("user", text);
  const thinkingId = addThinking();
  transcriptText.textContent = `"${text}"`;
  setStatus("processing", "Thinking...");

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
    setStatus("ready", `Turn ${data.turn}`);
    updateSentiment(data.sentiment);
    removeThinking(thinkingId);

    const isEscalation = data.action === "escalate";
    addMessage(isEscalation ? "escalated" : "agent", data.agent_response);
    if (isEscalation) escalationBanner.classList.remove("hidden");

    // Speak
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
    setStatus("error", "Error");
  }
}

// ── Process Voice Turn ───────────────────────────────────────────────────────
async function processVoiceTurn(audioBlob) {
  document.body.classList.add("has-messages");
  transcriptText.textContent = "Transcribing...";
  const thinkingId = addThinking();
  setStatus("processing", "Transcribing...");

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
    const ragSources = h.get("X-RAG-Sources");
    const toolUsed   = h.get("X-Tool-Used");

    if (newSession) sessionId = newSession;
    setStatus("ready", `Turn ${turn}`);

    transcriptText.textContent = transcript ? `"${transcript}"` : "";
    removeThinking(thinkingId);

    if (transcript) addMessage("user", transcript);
    updateSentiment({ label: sentLabel, score: sentScore });

    // Get agent text
    let agentText = "...";
    try {
      const convRes = await fetch(`${API_BASE}/agent/conversation/${sessionId}`);
      const conv = await convRes.json();
      agentText = conv.turns?.[conv.turns.length - 1]?.agent || agentText;
    } catch {}

    const isEscalation = action === "escalate";
    addMessage(isEscalation ? "escalated" : "agent", agentText);
    if (isEscalation) escalationBanner.classList.remove("hidden");

    // Play audio
    const audioBuffer = await res.arrayBuffer();
    const blob2 = new Blob([audioBuffer], { type: "audio/mpeg" });
    new Audio(URL.createObjectURL(blob2)).play().catch(() => {});

    updateDebug({ session_id: sessionId, turn, action, sentiment: { label: sentLabel, score: sentScore }, rag_sources: ragSources, tool_used: toolUsed });

  } catch (err) {
    removeThinking(thinkingId);
    addMessage("system", `Error: ${err.message}`);
    setStatus("error", "Error");
    transcriptText.textContent = "";
  }
}

// ── UI Helpers ───────────────────────────────────────────────────────────────
function addMessage(type, text) {
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = text;
  chatArea.appendChild(div);
  // Smooth scroll to the latest message
  requestAnimationFrame(() => {
    const main = document.querySelector(".main");
    main.scrollTo({ top: main.scrollHeight, behavior: "smooth" });
  });
  return div;
}

let thinkingCounter = 0;
function addThinking() {
  const id = `thinking-${thinkingCounter++}`;
  const div = document.createElement("div");
  div.className = "msg thinking"; div.id = id;
  div.innerHTML = '<div class="dot-loader"><span></span><span></span><span></span></div>';
  chatArea.appendChild(div);
  return id;
}
function removeThinking(id) { const el = document.getElementById(id); if (el) el.remove(); }

// ── Sentiment ────────────────────────────────────────────────────────────────
function updateSentiment(sentiment) {
  if (!sentiment) return;
  const { label, score } = sentiment;

  sentimentStrip.classList.remove("hidden");

  const emojiMap = { HAPPY: "😄", POSITIVE: "😊", CALM: "🙂", NEUTRAL: "😐", CONCERNED: "😕", FRUSTRATED: "😟", UPSET: "😤" };
  sentimentEmoji.textContent = emojiMap[label] || "😐";
  sentimentLabel.textContent = label || "NEUTRAL";
  sentimentScore.textContent = typeof score === "number" ? (score > 0 ? "+" : "") + score.toFixed(2) : "—";

  // Position thumb on track (score -1 to +1 → 0% to 100%)
  const pct = ((score + 1) / 2) * 100;
  sentimentThumb.style.left = `${pct}%`;

  // Fill bar
  sentimentFill.className = "sentiment-fill";
  if (score >= 0) {
    sentimentFill.style.left = "50%";
    sentimentFill.style.right = "auto";
    sentimentFill.style.width = `${(score / 1) * 50}%`;
  } else {
    sentimentFill.classList.add("neg");
    sentimentFill.style.left = "auto";
    sentimentFill.style.right = "50%";
    sentimentFill.style.width = `${(Math.abs(score) / 1) * 50}%`;
  }

  // Color
  const color = score > 0.2 ? "var(--positive)" : score < -0.2 ? "var(--negative)" : "var(--yellow)";
  sentimentThumb.style.borderColor = color;
  sentimentFill.style.background = color;
}

// ── Debug ─────────────────────────────────────────────────────────────────────
function updateDebug(data) {
  debugOutput.textContent = JSON.stringify({
    ...data,
    langsmith: "smith.langchain.com → voice-agent project",
  }, null, 2);
}
