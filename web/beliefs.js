// Read-only view of /api/beliefs. Both are behind the same basic-auth
// challenge, so by the time this script runs the browser already holds the
// credentials and sends them with the fetch automatically.

const beliefsEl = document.getElementById('beliefs');
const peopleEl = document.getElementById('people');
const questionsEl = document.getElementById('questions');
const countEl = document.getElementById('count');
const filterEl = document.getElementById('filter');

let data = null;

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderBeliefs(beliefs) {
  beliefsEl.innerHTML = '';
  if (!beliefs.length) {
    beliefsEl.appendChild(el('p', 'belief-empty', 'Nothing stored yet.'));
    return;
  }
  for (const b of beliefs) {
    const card = el('article', 'belief');
    const head = el('div', 'belief-head');
    head.appendChild(el('span', 'belief-statement', b.statement));
    head.appendChild(el('span', 'belief-status belief-' + b.status, b.status));
    head.appendChild(el('span', 'belief-confidence', b.confidence.toFixed(2)));
    card.appendChild(head);

    const list = el('ul', 'belief-evidence');
    if (!b.evidence.length) {
      list.appendChild(el('li', 'belief-none',
        'No evidence yet — generated during sleep, awaiting verification.'));
    }
    for (const e of b.evidence) {
      const parse = e.parse_confidence === null || e.parse_confidence === undefined
        ? '' : ` · parse ${e.parse_confidence.toFixed(2)}`;
      const li = el('li', 'belief-' + e.effect);
      li.appendChild(el('strong', null, e.person));
      const note = e.note ? ` · ${e.note}` : '';
      li.appendChild(document.createTextNode(` ${e.effect} it · ${e.at}${parse}${note}`));
      list.appendChild(li);
    }
    card.appendChild(list);
    beliefsEl.appendChild(card);
  }
}

function applyFilter() {
  if (!data) return;
  const q = filterEl.value.trim().toLowerCase();
  const shown = q
    ? data.beliefs.filter(b => b.statement.toLowerCase().includes(q)
        || b.evidence.some(e => e.person.toLowerCase().includes(q)))
    : data.beliefs;
  renderBeliefs(shown);
  countEl.textContent = q
    ? `${shown.length} of ${data.beliefs.length}`
    : `${data.beliefs.length} belief${data.beliefs.length === 1 ? '' : 's'}`;
}

async function load() {
  countEl.textContent = 'loading…';
  countEl.className = 'status-pill';
  try {
    const res = await fetch('/api/beliefs', { cache: 'no-store' });
    if (!res.ok) throw new Error('request failed: ' + res.status);
    data = await res.json();
  } catch (err) {
    countEl.textContent = 'unavailable';
    countEl.className = 'status-pill offline';
    beliefsEl.innerHTML = '';
    beliefsEl.appendChild(el('p', 'belief-empty', "Couldn't reach the server just now."));
    return;
  }
  countEl.className = 'status-pill online';
  applyFilter();

  peopleEl.innerHTML = '';
  document.getElementById('people-title').hidden = !data.people.length;
  for (const p of data.people) {
    const trust = Object.entries(p.trust)
      .map(([domain, score]) => `${domain} ${score.toFixed(2)}`).join(', ') || 'no trust recorded yet';
    const row = el('div', 'belief-person');
    row.appendChild(el('strong', null, p.name));
    row.appendChild(document.createTextNode(' — ' + trust));
    peopleEl.appendChild(row);
  }

  questionsEl.innerHTML = '';
  document.getElementById('questions-title').hidden = !data.open_questions.length;
  for (const q of data.open_questions) {
    const li = el('li');
    li.appendChild(el('span', 'belief-reason', q.reason));
    li.appendChild(document.createTextNode(' ' + q.question));
    questionsEl.appendChild(li);
  }
}

function download() {
  if (!data) return;
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:]/g, '-');
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `chloe-beliefs-${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

filterEl.addEventListener('input', applyFilter);
document.getElementById('refresh').addEventListener('click', load);
document.getElementById('download').addEventListener('click', download);
load();
