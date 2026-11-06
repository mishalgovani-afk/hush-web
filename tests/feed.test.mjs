// Parser tests for scripts/fetch-feed.mjs, using made-up sample posts in each source's format.
import assert from "node:assert/strict";
import { test } from "node:test";
import { htmlToText, merge, parseArchive, parseJsonSafeIds, parseMastodon, parseRss } from "../scripts/fetch-feed.mjs";

test("htmlToText keeps paragraphs, decodes entities, expands truncated links", () => {
  const html = '<p>Sample line one &amp; more</p><p>Second &quot;para&quot;<br>next line '
    + '<a href="https://example.com/a/long/path" rel="nofollow"><span class="invisible">https://</span><span class="ellipsis">example.com/a/lo</span><span class="invisible">ng/path</span></a></p>'
    + '<script>alert(1)</script>';
  assert.equal(htmlToText(html), 'Sample line one & more\n\nSecond "para"\nnext line https://example.com/a/long/path');
  assert.equal(htmlToText("<p>It&#39;s &#x1F600;</p>"), "It's 😀");
  assert.equal(htmlToText('<a href="https://x.test/u">@someone</a> hi'), "@someone hi");
});

test("parseMastodon: posts, reposts, media, counts", () => {
  const posts = parseMastodon([
    { id: "115000000000000002", created_at: "2026-11-05T12:00:00.000Z", url: "https://truthsocial.com/@realDonaldTrump/115000000000000002",
      content: "<p>Made-up sample post</p>", media_attachments: [{ type: "image", url: "https://static.example/img.jpg", preview_url: "https://static.example/p.jpg" }],
      replies_count: 5, reblogs_count: 6, favourites_count: 7, reblog: null },
    { id: "115000000000000001", created_at: "2026-11-05T11:00:00.000Z", url: null, content: "",
      reblog: { content: "<p>Shared text</p>", account: { acct: "someone" }, media_attachments: [], replies_count: 1, reblogs_count: 2, favourites_count: 3 } },
    { id: null, created_at: "x" },
  ]);
  assert.equal(posts.length, 2);
  assert.deepEqual(posts[0].media, [{ type: "image", url: "https://static.example/img.jpg", preview: "https://static.example/p.jpg" }]);
  assert.equal(posts[0].likes, 7);
  assert.equal(posts[1].repost, true);
  assert.equal(posts[1].repostOf, "@someone");
  assert.equal(posts[1].text, "Shared text");
  assert.equal(posts[1].url, "https://truthsocial.com/@realDonaldTrump/115000000000000001");
});

test("parseArchive accepts array or wrapped object, rejects unsafe URLs", () => {
  // Numeric 18-digit id, as raw JSON text: must survive parsing exactly.
  const rec = parseJsonSafeIds('{"id": 115000000000000003, "x": 1}');
  assert.equal(rec.id, "115000000000000003");
  Object.assign(rec, { created_at: "2026-11-04T10:00:00Z", content: "Archive sample", url: "javascript:alert(1)",
    media: ["https://static.example/v.mp4"], replies_count: "9" });
  for (const json of [[rec], { posts: [rec] }, { data: [rec] }]) {
    const [p] = parseArchive(json);
    assert.equal(p.id, "115000000000000003");
    assert.equal(p.url, "https://truthsocial.com/@realDonaldTrump/115000000000000003");
    assert.equal(p.media[0].type, "video");
    assert.equal(p.replies, 9);
  }
});

test("parseRss: trumpstruth-style items", () => {
  const xml = `<?xml version="1.0"?><rss xmlns:truth="https://truthsocial.com/ns"><channel><title>T</title>
    <item><title><![CDATA[[No Title] - Post from November 5, 2026]]></title>
      <link>https://trumpstruth.org/statuses/12345</link>
      <description><![CDATA[<p>RSS sample &amp; text</p>]]></description>
      <guid>https://trumpstruth.org/statuses/12345</guid>
      <pubDate>Wed, 05 Nov 2026 13:30:00 +0000</pubDate>
      <truth:originalUrl>https://truthsocial.com/@realDonaldTrump/115000000000000004</truth:originalUrl>
      <enclosure url="https://static.example/a.jpg" type="image/jpeg"/>
    </item>
    <item><title>Only a title here</title><link>https://truthsocial.com/@realDonaldTrump/115000000000000005</link>
      <description></description><pubDate>Wed, 05 Nov 2026 12:00:00 +0000</pubDate></item>
  </channel></rss>`;
  const [a, b] = parseRss(xml);
  assert.equal(a.id, "115000000000000004");
  assert.equal(a.text, "RSS sample & text");
  assert.equal(a.url, "https://truthsocial.com/@realDonaldTrump/115000000000000004");
  assert.equal(a.createdAt, "2026-11-05T13:30:00.000Z");
  assert.equal(a.media[0].url, "https://static.example/a.jpg");
  assert.equal(b.text, "Only a title here");
});

test("merge dedupes, sorts newest first, caps length", () => {
  const p = (id, t) => ({ id, createdAt: t, text: id, media: [] });
  const out = merge([p("3", "2026-11-03T00:00:00Z"), p("2", "2026-11-02T00:00:00Z")], [p("2", "2026-11-02T00:00:00Z"), p("1", "2026-11-01T00:00:00Z")], 2);
  assert.deepEqual(out.map((x) => x.id), ["3", "2"]);
});
