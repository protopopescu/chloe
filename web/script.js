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

// Chat — talks to this same server's /api/greet and /api/chat endpoints
// (see chloe-server.py). /api/greet is the friendly first-contact step: it turns
// the visitor's real typed name into a genuine ChloeEngine.greet() call,
// which (per the secret-word identity feature in chloe/dialogue.py) may
// itself ask to set up a secret word, or ask a returning name to confirm
// theirs before their message history is handed back to them. /api/chat
// handles every message after that, and if a vLLM server is configured,
// asks it to phrase the reply naturally.
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');
const chatLog = document.getElementById('chatLog');
const statusPill = document.getElementById('statusPill');
const sendButton = document.getElementById('chatSubmit');
const chatWho = document.getElementById('chatWho');
const chatWhoName = document.getElementById('chatWhoName');
const switchPersonBtn = document.getElementById('switchPerson');

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

// Whether the site is currently in a dreaming (chat-paused) window -- see
// chloe-server.py's DREAM_INTERVAL_MINUTES. Tracked so we only announce the
// start/end of a dream once, not on every poll while it's ongoing.
let wasDreaming = false;

function setDreaming(dreaming, summary) {
  chatInput.disabled = dreaming;
  sendButton.disabled = dreaming;
  if (dreaming) {
    if (!wasDreaming) {
      addMessage("💤 Chloe is dreaming — merging what she's learned, checking for contradictions, and preparing questions. Back shortly.", 'bot');
    }
  } else if (wasDreaming) {
    addMessage('🌅 Chloe woke up.' + (summary ? ' ' + summary : ''), 'bot');
    chatInput.focus();
  }
  wasDreaming = dreaming;
}

async function refreshStatus() {
  try {
    const res = await fetch('/api/health');
    const data = await res.json();
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
  chatInput.placeholder = 'Say something to CHLOE…';
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

switchPersonBtn.addEventListener('click', () => {
  // A fresh identity needs a fresh session_id too -- otherwise /api/greet
  // would just find the previous person's already-created engine for this
  // session_id and skip straight to "Welcome back" for the wrong name.
  localStorage.removeItem(NAME_KEY);
  localStorage.removeItem(SESSION_KEY);
  chatLog.innerHTML = '';
  addMessage("No problem — I'm CHLOE. What should I call you?", 'bot');
  enterNameMode();
});

async function initChat() {
  const storedName = getStoredName();
  chatLog.innerHTML = '';
  if (!storedName) {
    addMessage("Hi, I'm CHLOE. What should I call you?", 'bot');
    enterNameMode();
    return;
  }
  // Name already known for this browser (returning visit). Still call
  // /api/greet -- it's the only thing that either (a) creates the engine
  // for the first time this session_id has been seen server-side (e.g.
  // after a server restart, or on a device that already has the name
  // saved but never actually greeted), which can surface a secret-word
  // prompt, or (b) is a no-op "welcome back" if this session already
  // greeted. Either way the reply is safe to just show.
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
