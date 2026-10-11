// Parsing Server-Sent Events by hand. The backend sends messages like:
//
//     data: {"type":"delta","text":"Hel"}\n\n
//     data: {"type":"delta","text":"lo"}\n\n
//
// A network chunk can end in the middle of a message, so the unfinished tail is returned
// and must be put back in front of the next chunk.

export interface SseParseResult {
  payloads: string[]; // the text after "data: " for each complete message
  rest: string; // an unfinished message (or "") to prepend to the next chunk
}

export function parseSseChunk(buffer: string): SseParseResult {
  const blocks = buffer.split("\n\n");
  const rest = blocks.pop() ?? ""; // the last piece is either "" or an incomplete message

  const payloads: string[] = [];
  for (const block of blocks) {
    for (const line of block.split("\n")) {
      if (line.startsWith("data: ")) {
        payloads.push(line.slice("data: ".length));
      }
    }
  }
  return { payloads, rest };
}
