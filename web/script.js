// Mobile nav toggle
const navToggle = document.getElementById('navToggle');
const siteNav = document.getElementById('siteNav');

navToggle.addEventListener('click', () => {
  const open = siteNav.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(open));
});

siteNav.querySelectorAll('a').forEach(link => {
  link.addEventListener('click', () => {
    siteNav.classList.remove('open');
    navToggle.setAttribute('aria-expanded', 'false');
  });
});

// Chat — talks to this server's /api/greet and /api/chat (chloe-server.py).
// /api/greet turns the name the visitor typed into a ChloeEngine.greet()
// call, which may ask to set up a secret word, or ask a returning name to
// confirm theirs before their history is handed back. /api/chat handles
// every message after that.
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatLog = document.getElementById('chatLog');
const statusPill = document.getElementById('statusPill');
const sendButton = document.getElementById('chatSubmit');
const chatWho = document.getElementById('chatWho');
const chatWhoName = document.getElementById('chatWhoName');
const switchPersonBtn = document.getElementById('switchPerson');
const downloadBtn = document.getElementById('downloadChat');

const SESSION_KEY = 'chloe_session_id';
const NAME_KEY = 'chloe_name';

function getSessionId() {
  let id = localStorage.getItem(SESSION_KEY);
  if (!id) {
    id = crypto.randomUUID();
    localStorage.setItem(SESSION_KEY, id);
  }
  return id;
}

function getStoredName() {
  return localStorage.getItem(NAME_KEY);
}

function addMessage(text, from) {
  const div = document.createElement('div');
  div.className = 'msg ' + (from === 'user' ? 'msg-user' : 'msg-bot');
  div.textContent = text;
  chatLog.appendChild(div);
  chatLog.scrollTop = chatLog.scrollHeight;
  return div;
}

// Whether the site is in a dreaming (chat-paused) window. Tracked so the
// start and end of a dream are announced once, not on every poll.
let wasDreaming = false;

function setDreaming(dreaming, summary) {
  chatInput.disabled = dreaming;
  sendButton.disabled = dreaming;
  if (dreaming) {
    if (!wasDreaming) {
      addMessage("💤 Chloe is dreaming — merging what it has learned, checking for contradictions, and preparing questions. Back shortly.", 'bot');
    }
  } else if (wasDreaming) {
    addMessage('🌅 Chloe woke up.' + (summary ? ' ' + summary : ''), 'bot');
    chatInput.focus();
    askWhatSheThoughtOf();
  }
  wasDreaming = dreaming;
}

// Waking is where the questions consolidation produced get put to someone.
// Only a session that asked it to sleep is owed them, so this is safe to
// call on every wake -- the server returns nothing for everybody else.
async function askWhatSheThoughtOf() {
  try {
    const params = new URLSearchParams({ session_id: getSessionId(), name: getStoredName() || '' });
    const res = await fetch('/api/wake?' + params, { cache: 'no-store' });
    if (!res.ok) return;
    const data = await res.json();
    if (data.reply) addMessage(data.reply, 'bot');
  } catch (err) {
    /* it just stays quiet */
  }
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
    // The field's cap follows the parser's, so CHLOE_MAX_INPUT_CHARS is the
    // one place it is set. The attribute in index.html is the fallback for
    // before the first poll answers.
    if (data.max_input_chars) chatInput.maxLength = data.max_input_chars;
    if (data.dreaming) {
      statusPill.textContent = 'dreaming…';
      statusPill.className = 'status-pill dreaming';
    } else {
      statusPill.textContent = data.vllm_configured ? 'online' : 'engine only';
      statusPill.className = 'status-pill ' + (data.vllm_configured ? 'online' : '');
    }
    setDreaming(Boolean(data.dreaming), data.last_dream_summary);
  } catch (err) {
    statusPill.textContent = 'offline';
    statusPill.className = 'status-pill offline';
  }
}

// 'name' — the form is asking "what should I call you?" and the next
//          submission is treated as a name, not a chat message.
// 'chat' — normal conversation; submissions go to /api/chat.
let mode = 'name';

function enterNameMode() {
  mode = 'name';
  chatInput.placeholder = 'Your name…';
  chatInput.focus();
  sendButton.textContent = 'Introduce myself';
  chatWho.hidden = true;
}

function enterChatMode(name) {
  mode = 'chat';
  chatInput.placeholder = 'Say something to Chloe…';
  sendButton.textContent = 'Send';
  chatWhoName.textContent = name;
  chatWho.hidden = false;
}

async function submitName(name) {
  addMessage(name, 'user');
  const thinking = addMessage('…', 'bot');
  try {
    const res = await fetch('/api/greet', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: getSessionId(), name }),
    });
    if (!res.ok) throw new Error('request failed: ' + res.status);
    const data = await res.json();
    thinking.textContent = data.reply;
    localStorage.setItem(NAME_KEY, name);
    enterChatMode(name);
  } catch (err) {
    thinking.textContent = "I couldn't reach the server just now — try again in a moment.";
  }
}

async function submitChatMessage(text) {
  addMessage(text, 'user');
  const thinking = addMessage('…', 'bot');
  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: getSessionId(), name: getStoredName() || '', message: text }),
    });
    if (!res.ok) throw new Error('request failed: ' + res.status);
    const data = await res.json();
    thinking.textContent = data.reply;
  } catch (err) {
    thinking.textContent = "I couldn't reach the server just now — try again in a moment.";
  }
}

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (chatInput.disabled) return; // dreaming — see setDreaming()
  const text = chatInput.value.trim();
  if (!text) return;
  chatInput.value = '';
  sendButton.disabled = true;
  try {
    if (mode === 'name') {
      await submitName(text);
    } else {
      await submitChatMessage(text);
    }
  } finally {
    sendButton.disabled = false;
    chatLog.scrollTop = chatLog.scrollHeight;
  }
});

// Download the conversation. The server's copy is the provenance record --
// every turn as it was logged, with timestamps, plus the beliefs those turns
// produced and their current confidence -- so it survives page reloads and
// says more than the log on screen. If the server can't be reached, fall
// back to dumping what is visible, same as the chat itself falls back to the
// engine's own reply when the language model is unreachable.
function saveFile(text, filename, type) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function visibleTranscript() {
  return Array.from(chatLog.querySelectorAll('.msg')).map(node => ({
    role: node.classList.contains('msg-user') ? 'source' : 'chloe',
    text: node.textContent,
  }));
}

async function downloadChat() {
  const stamp = new Date().toISOString().slice(0, 19).replace(/:/g, '-');
  const name = getStoredName() || 'visitor';
  try {
    const params = new URLSearchParams({ session_id: getSessionId(), name: getStoredName() || '' });
    const res = await fetch('/api/transcript?' + params, { cache: 'no-store' });
    if (!res.ok) throw new Error('request failed: ' + res.status);
    const data = await res.json();
    if (!data.person) throw new Error('no server-side record yet');
    saveFile(JSON.stringify(data, null, 2),
             `chloe-chat-${name}-${stamp}.json`, 'application/json');
  } catch (err) {
    saveFile(JSON.stringify({
      exported_at: new Date().toISOString(),
      source: 'browser (server record unavailable: ' + err.message + ')',
      person: { name },
      turns: visibleTranscript(),
    }, null, 2), `chloe-chat-${name}-${stamp}.json`, 'application/json');
  }
}

downloadBtn.addEventListener('click', downloadChat);

switchPersonBtn.addEventListener('click', () => {
  // A fresh identity needs a fresh session_id: otherwise /api/greet finds
  // the previous person's engine for this session_id and goes straight to
  // "Welcome back" under the wrong name.
  localStorage.removeItem(NAME_KEY);
  localStorage.removeItem(SESSION_KEY);
  chatLog.innerHTML = '';
  addMessage("No problem — I'm Chloe. What should I call you?", 'bot');
  enterNameMode();
});

async function initChat() {
  const storedName = getStoredName();
  chatLog.innerHTML = '';
  if (!storedName) {
    addMessage("Hi, I'm Chloe. What should I call you?", 'bot');
    enterNameMode();
    return;
  }
  // Name already known for this browser. /api/greet is still called: it
  // either creates the engine for a session_id the server has not seen
  // (after a restart, or on a device holding a saved name that never
  // greeted), which can surface a secret-word prompt, or returns a "welcome
  // back" no-op. Either reply is safe to show.
  enterChatMode(storedName);
  const thinking = addMessage('…', 'bot');
  try {
    const res = await fetch('/api/greet', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: getSessionId(), name: storedName }),
    });
    if (!res.ok) throw new Error('request failed: ' + res.status);
    const data = await res.json();
    thinking.textContent = data.reply;
  } catch (err) {
    thinking.textContent = "I couldn't reach the server just now — try again in a moment.";
  }
}

initChat();
refreshStatus();
setInterval(refreshStatus, 5000); // catches dreaming state changes without a page reload
