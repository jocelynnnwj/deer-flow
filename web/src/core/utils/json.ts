import { parse } from "best-effort-json-parser";

export function parseJSON<T>(json: string | null | undefined, fallback: T) {
  if (!json) {
    return fallback;
  }
  try {
    let raw = json.trim()
      .replace(/^```js\s*/, "")
      .replace(/^```json\s*/, "")
      .replace(/^```ts\s*/, "")
      .replace(/^```plaintext\s*/, "")
      .replace(/^```\s*/, "")
      .replace(/\s*```$/, "");
    // Remove trailing commas before } or ]
    raw = raw.replace(/,\s*([}\]])/g, '$1');
    // Extract the first valid JSON object or array
    const match = raw.match(/([\[{].*[\]}])/s);
    if (match) {
      raw = match[1];
    }
    return JSON.parse(raw) as T;
  } catch (e) {
    console.error('parseJSON error, raw input:', json);
    return fallback;
  }
}
