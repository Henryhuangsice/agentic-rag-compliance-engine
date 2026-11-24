/** Parse the answer's small Markdown subset into data, so nothing is injected as HTML. */
export type Inline =
  | { type: "text"; text: string }
  | { type: "bold"; text: string }
  | { type: "cite"; index: number };
export type Block = { type: "p"; inline: Inline[] } | { type: "ul"; items: Inline[][] };

export function parseInline(text: string): Inline[] {
  return text
    .split(/(\*\*.+?\*\*|\[\d+\])/)
    .filter(Boolean)
    .map((part): Inline => {
      const bold = part.match(/^\*\*(.+)\*\*$/)?.[1];
      if (bold) return { type: "bold", text: bold };
      const cite = part.match(/^\[(\d+)\]$/)?.[1];
      if (cite) return { type: "cite", index: Number(cite) };
      return { type: "text", text: part };
    });
}

export function parseAnswer(text: string): Block[] {
  const blocks: Block[] = [];
  let list: Inline[][] | null = null;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    // The lookahead keeps a decimal such as "3.5 亿元" from reading as list item 3.
    const item = line.match(/^(?:[-•]\s*|\*(?!\*)\s*|\d+[.、)](?!\d)\s*)(.*)$/)?.[1];
    if (item !== undefined) {
      if (!list) blocks.push({ type: "ul", items: (list = []) });
      list.push(parseInline(item));
    } else {
      list = null;
      if (line) blocks.push({ type: "p", inline: parseInline(line.replace(/^#+\s*/, "")) });
    }
  }
  return blocks;
}
