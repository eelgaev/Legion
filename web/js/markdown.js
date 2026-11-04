// Markdown for assistant messages, with no third-party code. `parse` turns text into a plain
// object tree (pure, so node can test it without a DOM); the builder below turns that tree into
// DOM nodes with createElement/textContent/setAttribute only. Model text is never parsed as
// HTML, so raw HTML shows up as literal text and no sanitizer is needed. Links get an href only
// for http(s)/mailto, and images become plain links so the browser never fetches one.

const RENDER_INTERVAL_MS = 50;

const BLANK = /^\s*$/;
const FENCE = /^ {0,3}(`{3,}|~{3,})(.*)$/;
const ATX = /^ {0,3}(#{1,6}) +(.*)$/;
const HR = /^ {0,3}(-{3,}|\*{3,}|_{3,}) *$/;
const QUOTE = /^ {0,3}> ?(.*)$/;
const ITEM = /^( *)([-*+]|\d{1,9}[.)])( +)(.*)$/;
const DELIM_CELL = /^:?-+:?$/;
const PUNCT = /[!-/:-@[-`{-~]/;
const AUTOLINK = /^<([a-zA-Z][a-zA-Z0-9+.-]*:[^<>\s]*)>/;
// The destination may hold one level of balanced parentheses, e.g. a wiki link or alert(1).
const LINK = /^\[([^\]]*)\]\(((?:[^()\s]|\((?:[^()\s]*)\))*)\)/;
const IMAGE = /^!\[([^\]]*)\]\(((?:[^()\s]|\((?:[^()\s]*)\))*)\)/;
const BARE_URL = /^https?:\/\/[^\s<]+/;

const isAlnum = (c) => c !== undefined && /[A-Za-z0-9]/.test(c);
const isDigit = (c) => c !== undefined && c >= "0" && c <= "9";
const indentOf = (line) => line.match(/^ */)[0].length;

/** The href a link may use: http(s) or mailto, else null. Exported for tests. */
export function safeHref(url) {
  try {
    const u = new URL(String(url));
    return ["http:", "https:", "mailto:"].includes(u.protocol) ? u.href : null;
  } catch {
    return null;
  }
}

// ---------- inline parsing (pure) ----------

// The delimiter run that closes an emphasis, skipping escapes and code spans. `single` means
// the closer may not touch another of the same delimiter; `und` applies the rule that `_` is
// not emphasis inside a word (snake_case_name stays text). A run preceded by whitespace never
// closes, so `5 * 3 * 2` stays text.
function findClose(s, from, run, single, und) {
  for (let i = from; i < s.length; ) {
    const c = s[i];
    if (c === "\\") {
      i += 2;
      continue;
    }
    if (c === "`") {
      let n = 0;
      while (s[i + n] === "`") n++;
      const end = s.indexOf("`".repeat(n), i + n);
      if (end !== -1) {
        i = end + n;
        continue;
      }
    }
    if (s.startsWith(run, i)) {
      const before = s[i - 1];
      const after = s[i + run.length];
      if (/\s/.test(before) || (single && (before === run || after === run)) || (und && isAlnum(after))) {
        i++;
        continue;
      }
      return i;
    }
    i++;
  }
  return -1;
}

// A code span keeps its content verbatim, except one leading and trailing space are stripped.
const trimCodeSpaces = (t) =>
  t.length > 2 && t.startsWith(" ") && t.endsWith(" ") && t.trim() ? t.slice(1, -1) : t;

function parseInlines(s) {
  const out = [];
  let text = "";
  const flush = () => {
    if (text) {
      out.push({ type: "text", text });
      text = "";
    }
  };
  const push = (node) => {
    flush();
    out.push(node);
  };
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === "\\" && PUNCT.test(s[i + 1] ?? "")) {
      text += s[i + 1];
      i += 2;
      continue;
    }
    if (c === "\n") {
      push({ type: "br" }); // chat models rely on single newlines inside a paragraph
      i++;
      continue;
    }
    if (c === "`") {
      let n = 1;
      while (s[i + n] === "`") n++;
      const close = s.indexOf("`".repeat(n), i + n);
      if (close !== -1) {
        push({ type: "code", text: trimCodeSpaces(s.slice(i + n, close)) });
        i = close + n;
        continue;
      }
    }
    if (c === "!" || c === "[") {
      const m = (c === "!" ? IMAGE : LINK).exec(s.slice(i));
      if (m) {
        // Never an <img>: the browser would fetch the URL by itself, which lets an injected
        // prompt leak data. A link the user must click is safe.
        const inlines = c === "!" ? [{ type: "text", text: `[image: ${m[1]}]` }] : parseInlines(m[1]);
        push({ type: "link", href: m[2], inlines });
        i += m[0].length;
        continue;
      }
    }
    if (c === "<") {
      const m = AUTOLINK.exec(s.slice(i));
      if (m && safeHref(m[1])) {
        push({ type: "link", href: m[1], inlines: [{ type: "text", text: m[1] }] });
        i += m[0].length;
        continue;
      }
    }
    if (c === "h") {
      const m = BARE_URL.exec(s.slice(i));
      if (m) {
        let url = m[0];
        while (url && ".,;:!?)".includes(url[url.length - 1])) url = url.slice(0, -1);
        if (url) {
          push({ type: "link", href: url, inlines: [{ type: "text", text: url }] });
          i += url.length;
          continue;
        }
      }
    }
    if (c === "*" || c === "_" || (c === "~" && s[i + 1] === "~")) {
      const dbl = s[i + 1] === c;
      const run = dbl ? c + c : c;
      // A run followed by whitespace never opens, and `_` never opens inside a word.
      if (!/\s/.test(s[i + run.length] ?? "") && !(c === "_" && isAlnum(s[i - 1]))) {
        const close = findClose(s, i + run.length, run, !dbl, c === "_");
        if (close !== -1) {
          const kind = c === "~" ? "del" : dbl ? "strong" : "em";
          push({ type: kind, inlines: parseInlines(s.slice(i + run.length, close)) });
          i = close + run.length;
          continue;
        }
      }
    }
    text += c;
    i++;
  }
  flush();
  return out;
}

// ---------- block parsing (pure) ----------

/** Markdown to a plain block tree. Exported for tests; the builder below renders it. */
export function parse(text) {
  return parseBlocks(String(text).replace(/\r\n?/g, "\n").split("\n"));
}

const startsBlock = (lines, i) => {
  const l = lines[i];
  return FENCE.test(l) || ATX.test(l) || HR.test(l) || QUOTE.test(l) || ITEM.test(l) || isTable(lines, i);
};

function parseBlocks(lines) {
  const blocks = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (BLANK.test(line)) {
      i++;
      continue;
    }
    let m;
    if ((m = FENCE.exec(line))) {
      // Closed by the same character at the same or greater length; an unclosed fence runs to
      // the end of the text, so a half-received fence still renders while streaming.
      const close = new RegExp("^ {0,3}" + m[1][0] + "{" + m[1].length + ",} *$");
      const body = [];
      i++;
      while (i < lines.length && !close.test(lines[i])) body.push(lines[i++]);
      const unclosed = i >= lines.length;
      i++; // the closing fence, or past the end
      if (unclosed) while (body.length && BLANK.test(body.at(-1))) body.pop();
      blocks.push({ type: "code", lang: m[2].trim().split(/\s+/)[0], text: body.join("\n") });
      continue;
    }
    if ((m = ATX.exec(line))) {
      blocks.push({ type: "heading", level: m[1].length, inlines: parseInlines(m[2].trim()) });
      i++;
      continue;
    }
    if (HR.test(line)) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }
    if (QUOTE.test(line)) {
      const inner = [];
      while (i < lines.length && QUOTE.test(lines[i])) {
        inner.push(QUOTE.exec(lines[i])[1]);
        i++;
      }
      blocks.push({ type: "quote", blocks: parseBlocks(inner) });
      continue;
    }
    if (isTable(lines, i)) {
      const [table, next] = parseTable(lines, i);
      blocks.push(table);
      i = next;
      continue;
    }
    if (ITEM.test(line)) {
      const [list, next] = parseList(lines, i);
      blocks.push(list);
      i = next;
      continue;
    }
    const para = [];
    while (i < lines.length && !BLANK.test(lines[i]) && !startsBlock(lines, i)) para.push(lines[i++]);
    blocks.push({ type: "paragraph", inlines: parseInlines(para.join("\n")) });
  }
  return blocks;
}

// Items start at the same indent as the first one; lines indented to the item's content column
// belong to that item, so a 2+ space deeper marker nests. Task boxes stay plain text.
function parseList(lines, start) {
  const first = ITEM.exec(lines[start]);
  const ordered = isDigit(first[2][0]);
  const base = first[1].length;
  const list = { type: "list", ordered, start: ordered ? Number(first[2].slice(0, -1)) : 1, items: [] };
  let i = start;
  while (i < lines.length) {
    const m = ITEM.exec(lines[i]);
    if (!m || m[1].length !== base || isDigit(m[2][0]) !== ordered) break;
    const contentCol = m[1].length + m[2].length + Math.min(m[3].length, 4);
    const item = [m[4]];
    i++;
    while (i < lines.length) {
      if (BLANK.test(lines[i])) {
        let j = i;
        while (j < lines.length && BLANK.test(lines[j])) j++;
        if (j >= lines.length || indentOf(lines[j]) < contentCol) break;
        while (i < j) item.push(lines[i++]);
      } else if (indentOf(lines[i]) >= contentCol) {
        item.push(lines[i++].slice(contentCol));
      } else break;
    }
    list.items.push(parseBlocks(item));
    let j = i; // a blank line keeps the list going when another item follows at this level
    while (j < lines.length && BLANK.test(lines[j])) j++;
    const next = j < lines.length ? ITEM.exec(lines[j]) : null;
    if (next && next[1].length === base && isDigit(next[2][0]) === ordered) i = j;
  }
  return [list, i];
}

// A table is a header row plus a delimiter row of `:---`-style cells, both with the same number
// of (unescaped) pipe-separated cells.
function isTable(lines, i) {
  if (i + 1 >= lines.length || !lines[i].includes("|") || !lines[i + 1].includes("|")) return false;
  const head = splitRow(lines[i]);
  const delim = splitRow(lines[i + 1]);
  return head.length === delim.length && delim.every((c) => DELIM_CELL.test(c));
}

const alignOf = (c) =>
  c.startsWith(":") ? (c.endsWith(":") ? "center" : "left") : c.endsWith(":") ? "right" : null;

function parseTable(lines, i) {
  const align = splitRow(lines[i + 1]).map(alignOf);
  const head = splitRow(lines[i]).map((c) => parseInlines(c));
  const rows = [];
  i += 2;
  while (i < lines.length && !BLANK.test(lines[i]) && lines[i].includes("|")) {
    const cells = splitRow(lines[i]).map((c) => parseInlines(c));
    while (cells.length < align.length) cells.push([]); // pad short rows
    rows.push(cells.slice(0, align.length)); // drop extra cells
    i++;
  }
  return [{ type: "table", align, head, rows }, i];
}

// Cells split on unescaped pipes; `\|` stays a literal pipe in the cell.
function splitRow(line) {
  let s = line.trim();
  if (s.startsWith("|")) s = s.slice(1);
  if (s.endsWith("|") && !s.endsWith("\\|")) s = s.slice(0, -1);
  const cells = [""];
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "\\" && s[i + 1] === "|") {
      cells[cells.length - 1] += "|";
      i++;
    } else if (s[i] === "|") cells.push("");
    else cells[cells.length - 1] += s[i];
  }
  return cells.map((c) => c.trim());
}

// ---------- DOM building (browser only) ----------

const make = (tag, attrs = {}, kids = []) => {
  const node = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) node.setAttribute(name, value);
  node.append(...kids);
  return node;
};

const textNode = (s) => document.createTextNode(s);

function inlineNodes(inlines) {
  const out = [];
  for (const n of inlines) {
    if (n.type === "text") out.push(textNode(n.text));
    else if (n.type === "code") out.push(make("code", {}, [textNode(n.text)]));
    else if (n.type === "br") out.push(make("br"));
    else if (n.type === "link") {
      const href = safeHref(n.href);
      const kids = inlineNodes(n.inlines);
      // Any other scheme, or an unparseable URL, loses the <a>: the text stays plain.
      out.push(...(href ? [make("a", { href, target: "_blank", rel: "noopener noreferrer" }, kids)] : kids));
    } else out.push(make(n.type, {}, inlineNodes(n.inlines))); // em, strong, del
  }
  return out;
}

function blockNode(b) {
  switch (b.type) {
    case "heading":
      return make(`h${b.level}`, {}, inlineNodes(b.inlines));
    case "code": {
      const code = make("code");
      code.textContent = b.text; // the code is text, never markup
      return make("div", { class: "code-block" }, [
        make("div", { class: "code-head" }, [
          make("span", {}, [textNode(b.lang || "text")]),
          make("button", { type: "button", class: "copy-code" }, [textNode("Copy")]),
        ]),
        make("pre", {}, [code]),
      ]);
    }
    case "list": {
      const items = b.items.map((blocks) =>
        blocks.length === 1 && blocks[0].type === "paragraph"
          ? make("li", {}, inlineNodes(blocks[0].inlines)) // a tight item holds its text directly
          : make("li", {}, blocks.map(blockNode)),
      );
      return make(b.ordered ? "ol" : "ul", b.ordered ? { start: String(b.start) } : {}, items);
    }
    case "quote":
      return make("blockquote", {}, b.blocks.map(blockNode));
    case "table": {
      const cell = (inlines, col, tag) =>
        make(tag, b.align[col] ? { style: `text-align:${b.align[col]}` } : {}, inlineNodes(inlines));
      return make("table", {}, [
        make("thead", {}, [make("tr", {}, b.head.map((c, j) => cell(c, j, "th")))]),
        make("tbody", {}, b.rows.map((r) => make("tr", {}, r.map((c, j) => cell(c, j, "td"))))),
      ]);
    }
    case "hr":
      return make("hr");
    default:
      return make("p", {}, inlineNodes(b.inlines));
  }
}

/** Sets `el` to the rendered markdown of `text`. */
export function setMarkdown(el, text) {
  el.replaceChildren(...parse(text).map(blockNode));
}

/**
 * Re-renders `el` at most once per ~50 ms while text streams in.
 * `update(text)` schedules a render, `flush()` renders the latest text now.
 */
export function streamInto(el, onRender = () => {}) {
  let latest = "";
  let last = 0;
  let timer = 0;
  const run = () => {
    timer = 0;
    last = performance.now();
    setMarkdown(el, latest);
    onRender();
  };
  return {
    update(text) {
      latest = text;
      if (!timer) timer = setTimeout(run, Math.max(0, RENDER_INTERVAL_MS - (performance.now() - last)));
    },
    flush() {
      clearTimeout(timer);
      run();
    },
  };
}

/** Wires the Copy buttons inside `root` (one delegated listener, survives re-renders). */
export function initMarkdown(root) {
  root.addEventListener("click", async (e) => {
    const button = e.target.closest?.("button.copy-code");
    if (!button) return;
    const code = button.closest(".code-block")?.querySelector("code");
    if (!code) return;
    try {
      await navigator.clipboard.writeText(code.textContent);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Failed";
    }
    setTimeout(() => {
      if (button.isConnected) button.textContent = "Copy";
    }, 1500);
  });
}
