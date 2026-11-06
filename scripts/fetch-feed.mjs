// Fetch Donald Trump's latest Truth Social posts and write a normalised, plain-text feed.json.
// Runs on GitHub Actions (see .github/workflows/feed.yml). No dependencies: Node 22+.
//
// Usage: node scripts/fetch-feed.mjs <previous feed.json or ""> <output path>
//
// Sources are tried in order; the first that returns posts wins. Posts are merged with the
// previous feed so the history survives a source hiccup.

import { readFileSync, writeFileSync, existsSync } from "node:fs";

export const ACCOUNT_ID = "107780257626128497"; // @realDonaldTrump
const UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1";
const KEEP = 300;

// ------------------------------------------------------------------------------------ text helpers

const ENTITIES = { amp: "&", lt: "<", gt: ">", quot: "\"", apos: "'", nbsp: " ", "#39": "'" };

export function decodeEntities(s) {
  return String(s ?? "").replace(/&(#x[0-9a-f]+|#\d+|[a-z]+\d*);/gi, (m, e) => {
    if (e[0] === "#") {
      const n = e[1] === "x" || e[1] === "X" ? parseInt(e.slice(2), 16) : parseInt(e.slice(1), 10);
      return Number.isFinite(n) ? String.fromCodePoint(n) : m;
    }
    return ENTITIES[e.toLowerCase()] ?? m;
  });
}

/** HTML -> plain text with paragraph/line breaks kept. Links become their visible text. */
export function htmlToText(html) {
  let s = String(html ?? "");
  s = s.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1");
  s = s.replace(/<(script|style)[\s\S]*?<\/\1>/gi, "");
  s = s.replace(/<br\s*\/?>/gi, "\n").replace(/<\/p>\s*<p[^>]*>/gi, "\n\n").replace(/<\/?(p|div)[^>]*>/gi, "\n");
  // Mastodon hides the scheme/tail of links in <span class="invisible">; keep the full URL.
  s = s.replace(/<a [^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/gi, (m, href, inner) => {
    const text = decodeEntities(inner.replace(/<[^>]+>/g, ""));
    return /^https?:\/\//.test(text) || text.includes("…") || /class="invisible"/.test(inner) ? decodeEntities(href) : text;
  });
  s = s.replace(/<[^>]+>/g, "");
  s = decodeEntities(s);
  return s.replace(/\r/g, "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
}

const safeUrl = (u) => (typeof u === "string" && /^https:\/\//i.test(u) ? u : null);
const isoDate = (d) => { const t = new Date(d); return Number.isNaN(t.getTime()) ? null : t.toISOString(); };
const num = (v) => (Number.isFinite(+v) ? +v : null);

function media(list) {
  return (Array.isArray(list) ? list : [])
    .map((m) => (typeof m === "string" ? { type: /\.(mp4|mov|m3u8)(\?|$)/i.test(m) ? "video" : "image", url: m } : m))
    .map((m) => ({
      type: ["video", "gifv"].includes(m.type) ? "video" : "image",
      url: safeUrl(m.url ?? m.remote_url),
      preview: safeUrl(m.preview_url ?? m.preview ?? m.thumbnail) ?? null,
    }))
    .filter((m) => m.url || m.preview)
    .slice(0, 4);
}

function post(p) {
  if (!p.id || !p.createdAt) return null;
  return {
    id: String(p.id),
    createdAt: p.createdAt,
    text: p.text ?? "",
    url: safeUrl(p.url) ?? `https://truthsocial.com/@realDonaldTrump/${p.id}`,
    repost: Boolean(p.repost),
    repostOf: p.repostOf ?? null,
    media: p.media ?? [],
    replies: p.replies ?? null,
    reposts: p.reposts ?? null,
    likes: p.likes ?? null,
  };
}

// ------------------------------------------------------------------------------------ parsers

/** Mastodon-style status objects (Truth Social's own API). */
export function parseMastodon(json) {
  const list = Array.isArray(json) ? json : [];
  return list.map((s) => {
    const r = s.reblog;
    const src = r ?? s;
    return post({
      id: s.id,
      createdAt: isoDate(s.created_at),
      text: htmlToText(src.content) || (src.media_attachments?.length ? "" : ""),
      url: s.url ?? src.url,
      repost: Boolean(r),
      repostOf: r?.account?.acct ? `@${r.account.acct}` : null,
      media: media(src.media_attachments),
      replies: num(src.replies_count), reposts: num(src.reblogs_count), likes: num(src.favourites_count),
    });
  }).filter(Boolean);
}

/** JSON.parse loses precision on 18-digit numeric ids; quote them first. */
export function parseJsonSafeIds(text) {
  return JSON.parse(String(text).replace(/"(id|in_reply_to_id)"\s*:\s*(\d{16,})/g, '"$1":"$2"'));
}

/** CNN's archive: an array (or {posts|data:[...]}) of {id, created_at, content, url, media, *_count}. */
export function parseArchive(json) {
  const list = Array.isArray(json) ? json : json?.posts ?? json?.data ?? json?.truths ?? [];
  return list.map((s) => post({
    id: s.id,
    createdAt: isoDate(s.created_at ?? s.createdAt ?? s.date),
    text: htmlToText(s.content ?? s.text ?? ""),
    url: s.url,
    repost: Boolean(s.reblog || s.is_reblog || s.repost),
    media: media(s.media ?? s.media_attachments),
    replies: num(s.replies_count), reposts: num(s.reblogs_count), likes: num(s.favourites_count),
  })).filter(Boolean);
}

/** RSS 2.0 (trumpstruth.org). */
export function parseRss(xml) {
  const items = String(xml).match(/<item\b[\s\S]*?<\/item>/gi) ?? [];
  const tag = (block, name) => {
    const m = block.match(new RegExp(`<${name}\\b[^>]*>([\\s\\S]*?)<\\/${name}>`, "i"));
    return m ? m[1].replace(/^\s*<!\[CDATA\[([\s\S]*?)\]\]>\s*$/, "$1").trim() : null;
  };
  return items.map((it) => {
    const original = tag(it, "truth:originalUrl") ?? tag(it, "link");
    const idMatch = (tag(it, "truth:originalId") ?? original ?? tag(it, "guid") ?? "").match(/(\d{12,})/);
    const enclosures = [...it.matchAll(/<(?:enclosure|media:content)\b[^>]*url="([^"]+)"[^>]*>/gi)].map((m) => decodeEntities(m[1]));
    const body = tag(it, "content:encoded") ?? tag(it, "description") ?? "";
    let text = htmlToText(body);
    const title = htmlToText(tag(it, "title") ?? "");
    if (!text && title && !/^\[No Title\]/i.test(title)) text = title;
    return post({
      id: idMatch?.[1],
      createdAt: isoDate(tag(it, "pubDate") ?? tag(it, "dc:date")),
      text,
      url: decodeEntities(original ?? ""),
      repost: /^RT[: ]/.test(text),
      media: media(enclosures),
    });
  }).filter(Boolean);
}

export function merge(fresh, previous, keep = KEEP) {
  const byId = new Map();
  for (const p of previous ?? []) byId.set(p.id, p);
  for (const p of fresh) byId.set(p.id, { ...byId.get(p.id), ...p });
  return [...byId.values()].sort((a, b) => b.createdAt.localeCompare(a.createdAt) || b.id.localeCompare(a.id)).slice(0, keep);
}

// ------------------------------------------------------------------------------------ sources

async function get(url, accept) {
  const res = await fetch(url, { headers: { "User-Agent": UA, Accept: accept }, signal: AbortSignal.timeout(45000) });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res;
}

export const SOURCES = [
  {
    name: "truthsocial.com",
    async load() {
      const res = await get(`https://truthsocial.com/api/v1/accounts/${ACCOUNT_ID}/statuses?limit=40&exclude_replies=true`, "application/json");
      return parseMastodon(parseJsonSafeIds(await res.text()));
    },
  },
  {
    name: "CNN Truth Social archive",
    async load() {
      const res = await get("https://ix.cnn.io/data/truth-social/truth_archive.json", "application/json");
      return parseArchive(parseJsonSafeIds(await res.text())).sort((a, b) => b.createdAt.localeCompare(a.createdAt)).slice(0, KEEP);
    },
  },
  {
    name: "trumpstruth.org",
    async load() {
      const res = await get("https://trumpstruth.org/feed", "application/rss+xml, application/xml, text/xml");
      return parseRss(await res.text());
    },
  },
];

async function main() {
  const [prevPath, outPath = "feed.json"] = process.argv.slice(2);
  let previous = { posts: [] };
  if (prevPath && existsSync(prevPath)) {
    try { previous = JSON.parse(readFileSync(prevPath, "utf8")); } catch { console.log("Previous feed unreadable, starting fresh"); }
  }
  const attempts = [];
  let fresh = null, used = null;
  for (const s of SOURCES) {
    try {
      const posts = await s.load();
      const withText = posts.filter((p) => p.text || p.media.length).length;
      console.log(`${s.name}: ${posts.length} posts (${withText} with content), newest ${posts[0]?.createdAt ?? "-"}`);
      attempts.push({ source: s.name, ok: posts.length > 0, posts: posts.length });
      if (posts.length) { fresh = posts; used = s.name; break; }
    } catch (e) {
      console.log(`${s.name}: failed (${e.message})`);
      attempts.push({ source: s.name, ok: false, error: e.message });
    }
  }
  const now = new Date().toISOString();
  const out = {
    account: "@realDonaldTrump",
    updatedAt: fresh ? now : previous.updatedAt ?? null,
    checkedAt: now,
    source: used ?? previous.source ?? null,
    attempts,
    posts: merge(fresh ?? [], previous.posts ?? []),
  };
  writeFileSync(outPath, JSON.stringify(out));
  console.log(`Wrote ${out.posts.length} posts from ${used ?? "previous feed (all sources failed)"}`);
  if (!fresh && !out.posts.length) process.exitCode = 1;
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
