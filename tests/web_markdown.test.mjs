// Run with: node --test tests/web_markdown.test.mjs
// Only the pure parts of markdown.js (parse, safeHref) are tested here: node has no DOM, so
// we assert on the block tree, not on rendered elements.
import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parse, safeHref } from "../web/js/markdown.js";

const WEB = path.join(path.dirname(fileURLToPath(import.meta.url)), "..", "web");
const readWeb = (...parts) => readFileSync(path.join(WEB, ...parts), "utf8");

test("ATX headings", () => {
  const blocks = parse("# One\n###### Six\n## Setup steps\n");
  assert.deepEqual(blocks.map((b) => b.level), [1, 6, 2]);
  assert.deepEqual(blocks[2].inlines, [{ type: "text", text: "Setup steps" }]);
  const [p] = parse("#not-a-heading"); // the hash needs a space after it
  assert.equal(p.type, "paragraph");
});

test("a paragraph keeps single newlines as br", () => {
  const [p] = parse("line one\nline two\n\nnext paragraph");
  assert.deepEqual(p.inlines, [
    { type: "text", text: "line one" },
    { type: "br" },
    { type: "text", text: "line two" },
  ]);
  assert.equal(parse("line one\nline two\n\nnext paragraph")[1].inlines[0].text, "next paragraph");
});

test("nested bullet lists with a code fence inside an item", () => {
  const [list] = parse("- outer\n  - inner\n    ```sh\n    echo hi\n    ```\n- next\n");
  assert.equal(list.type, "list");
  assert.equal(list.ordered, false);
  assert.equal(list.items.length, 2);
  const [outerPara, nested] = list.items[0];
  assert.equal(outerPara.type, "paragraph");
  assert.equal(outerPara.inlines[0].text, "outer");
  assert.equal(nested.type, "list");
  assert.equal(nested.items.length, 1);
  const [innerPara, code] = nested.items[0];
  assert.equal(innerPara.inlines[0].text, "inner");
  assert.equal(code.type, "code");
  assert.equal(code.lang, "sh");
  assert.equal(code.text, "echo hi");
  assert.equal(list.items[1][0].inlines[0].text, "next");
});

test("ordered lists keep the start number", () => {
  const [list] = parse("3. three\n4. four\n");
  assert.equal(list.ordered, true);
  assert.equal(list.start, 3);
  assert.equal(list.items.length, 2);
  const [paren] = parse("7) seven\n8) eight");
  assert.equal(paren.start, 7);
});

test("tables: alignment, escaped pipes, padding and dropping cells", () => {
  const [t] = parse(
    "| left | mid | right | pipe \\| here |\n|:-----|:---:|------:|-----|\n| a | b |\n| 1 | 2 | 3 | 4 | 5 |\n",
  );
  assert.equal(t.type, "table");
  assert.deepEqual(t.align, ["left", "center", "right", null]);
  assert.equal(t.head.length, 4);
  assert.deepEqual(t.head[3], [{ type: "text", text: "pipe | here" }]);
  assert.equal(t.rows.length, 2);
  assert.deepEqual(t.rows[0][1], [{ type: "text", text: "b" }]);
  assert.deepEqual(t.rows[0][3], []); // too few cells: padded
  assert.equal(t.rows[1].length, 4); // too many cells: extras dropped
});

test("an unclosed fence runs to the end of the text (streaming)", () => {
  const blocks = parse("intro\n\n```python\nprint('hi')\n");
  assert.equal(blocks.length, 2);
  assert.equal(blocks[1].type, "code");
  assert.equal(blocks[1].lang, "python");
  assert.equal(blocks[1].text, "print('hi')");
});

test("underscores inside a word are not emphasis", () => {
  const [p] = parse("snake_case_name");
  assert.deepEqual(p.inlines, [{ type: "text", text: "snake_case_name" }]);
});

test("**bold *both* bold** nests the em inside the strong", () => {
  const [p] = parse("**bold *both* bold**");
  assert.equal(p.inlines.length, 1);
  assert.deepEqual(p.inlines[0], {
    type: "strong",
    inlines: [
      { type: "text", text: "bold " },
      { type: "em", inlines: [{ type: "text", text: "both" }] },
      { type: "text", text: " bold" },
    ],
  });
});

test("a code span keeps its stars", () => {
  const [p] = parse("use `a * b` here");
  assert.deepEqual(p.inlines, [
    { type: "text", text: "use " },
    { type: "code", text: "a * b" },
    { type: "text", text: " here" },
  ]);
});

test("an image becomes a link with [image: alt] text", () => {
  const [p] = parse("![diagram](https://example.com/a.png)");
  assert.deepEqual(p.inlines, [
    {
      type: "link",
      href: "https://example.com/a.png",
      inlines: [{ type: "text", text: "[image: diagram]" }],
    },
  ]);
});

test("raw HTML stays text", () => {
  const [p] = parse("<script>alert(1)</script>");
  assert.deepEqual(p.inlines, [{ type: "text", text: "<script>alert(1)</script>" }]);
});

test("bare URLs stop at whitespace and drop trailing punctuation", () => {
  const [p] = parse("see https://example.com/a. then");
  assert.deepEqual(p.inlines, [
    { type: "text", text: "see " },
    {
      type: "link",
      href: "https://example.com/a",
      inlines: [{ type: "text", text: "https://example.com/a" }],
    },
    { type: "text", text: ". then" },
  ]);
});

test("backslash escapes keep punctuation literal", () => {
  const [p] = parse("a \\*not em\\* b");
  assert.deepEqual(p.inlines, [{ type: "text", text: "a *not em* b" }]);
});

test("blockquotes nest and contain any blocks", () => {
  const [q] = parse("> noted\n>\n> ```\n> x = 1\n> ```\n> > deep");
  assert.equal(q.type, "quote");
  assert.deepEqual(q.blocks.map((b) => b.type), ["paragraph", "code", "quote"]);
  assert.equal(q.blocks[2].blocks[0].inlines[0].text, "deep");
});

test("hr, autolinks and unmatched delimiters", () => {
  assert.deepEqual(parse("---\n***\n___"), [{ type: "hr" }, { type: "hr" }, { type: "hr" }]);
  const [p] = parse("<https://example.com> and 5 * 3 * 2");
  assert.equal(p.inlines[0].type, "link");
  assert.equal(p.inlines[0].href, "https://example.com");
  const [stars] = parse("a * b");
  assert.deepEqual(stars.inlines, [{ type: "text", text: "a * b" }]); // unmatched: literal
});

test("safeHref only allows http, https and mailto", () => {
  assert.equal(safeHref("javascript:alert(1)"), null);
  assert.equal(safeHref("JaVaScRiPt:x"), null);
  assert.equal(safeHref("data:text/html,x"), null);
  assert.equal(safeHref("https://a.b/c"), "https://a.b/c");
  assert.equal(safeHref("mailto:a@b"), "mailto:a@b");
  assert.equal(safeHref("foo"), null); // relative URLs are not links either
});

// Hard rule: model text is built into DOM nodes, never parsed as HTML, so the string-to-HTML
// APIs must not appear anywhere in web/js/.
test("web/js never touches the DOM-HTML APIs", () => {
  const forbidden = [
    "innerHTML",
    "outerHTML",
    "insertAdjacentHTML",
    "DOMParser",
    "document.write",
    "createContextualFragment",
  ];
  const dir = path.join(WEB, "js");
  for (const name of readdirSync(dir)) {
    if (!name.endsWith(".js")) continue;
    const src = readWeb("js", name);
    for (const api of forbidden) assert.ok(!src.includes(api), `web/js/${name} uses ${api}`);
  }
});

test("web/ ships no third-party code and markdown.js stays small", () => {
  assert.throws(() => readdirSync(path.join(WEB, "vendor")));
  const lines = readWeb("js", "markdown.js").split("\n").length;
  assert.ok(lines <= 450, `markdown.js is ${lines} lines`);
});
