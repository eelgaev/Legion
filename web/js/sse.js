// Parses a server-sent-events stream into the JSON payloads of its `data:` lines.
// Input: an async iterable of text chunks (chunks may split lines and events anywhere).
// Output: async generator of parsed JSON values; stops at `data: [DONE]`.
export async function* parseSse(chunks) {
  let buffer = "";
  let data = []; // data lines of the event being read
  const finish = function* () {
    if (data.length === 0) return false;
    const payload = data.join("\n");
    data = [];
    if (payload === "[DONE]") return true;
    yield JSON.parse(payload);
    return false;
  };
  for await (const chunk of chunks) {
    buffer += chunk;
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).replace(/\r$/, "");
      buffer = buffer.slice(nl + 1);
      if (line === "") {
        if (yield* finish()) return;
      } else if (line.startsWith("data:")) {
        data.push(line.slice(5).replace(/^ /, ""));
      } // other fields and ":" comments are ignored
    }
  }
  // Stream ended without a trailing blank line: flush what is left.
  if (buffer.startsWith("data:")) data.push(buffer.slice(5).replace(/^ /, ""));
  yield* finish();
}
