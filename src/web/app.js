const form = document.querySelector("#search-form");
const query = document.querySelector("#query");
const submit = document.querySelector("#submit");
const status = document.querySelector("#status");
const results = document.querySelector("#results");
const summary = document.querySelector("#summary");
const emptyState = document.querySelector("#empty-state");
const template = document.querySelector("#result-template");

function formatTime(timestampUs) {
  const seconds = Math.floor(timestampUs / 1_000_000);
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(seconds % 60).padStart(2, "0")}`;
}

function videoUrl(watchUrl, timestampUs) {
  const url = new URL(watchUrl);
  if (!["https:", "http:"].includes(url.protocol)) throw new Error("Invalid video URL");
  if (/(^|\.)youtube\.com$/.test(url.hostname) || url.hostname === "youtu.be") {
    url.searchParams.delete("start");
    url.searchParams.set("t", Math.floor(timestampUs / 1_000_000));
  }
  return url.href;
}

function renderResults(hits) {
  const fragment = document.createDocumentFragment();
  hits.forEach((hit, index) => {
    const card = template.content.cloneNode(true);
    const image = card.querySelector("img");
    const preview = card.querySelector(".preview");
    const watch = card.querySelector(".watch-link");
    const timestamp = formatTime(hit.timestamp_us);
    const url = videoUrl(hit.watch_url, hit.timestamp_us);
    preview.href = watch.href = url;
    preview.setAttribute("aria-label", `Open ${hit.video_id} at ${timestamp}`);
    image.alt = `${hit.video_id} at ${timestamp}`;
    image.src = `/frames/${encodeURIComponent(hit.video_id)}/${hit.frame_idx}.jpg`;
    image.addEventListener("error", () => {
      image.hidden = true;
      preview.querySelector(".image-error").hidden = false;
    }, { once: true });
    card.querySelector(".rank").textContent = `#${index + 1}`;
    card.querySelector(".timestamp").textContent = timestamp;
    card.querySelector("h3").textContent = hit.video_id;
    card.querySelector(".frame-index").textContent = `Frame ${hit.frame_idx}`;
    card.querySelector(".score").textContent = `Similarity ${Number(hit.score).toFixed(3)}`;
    fragment.append(card);
  });
  results.replaceChildren(fragment);
}

function setLoading(loading) {
  submit.disabled = loading;
  submit.querySelector(".spinner").hidden = !loading;
  document.querySelector("#button-label").textContent = loading ? "Searching…" : "Search →";
  results.setAttribute("aria-busy", String(loading));
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (submit.disabled) return;
  const text = query.value.trim();
  if (!text) { query.focus(); return; }
  const start = performance.now();
  setLoading(true);
  status.classList.remove("error");
  status.textContent = "Searching… The first search may take a little longer.";
  emptyState.hidden = true;
  results.replaceChildren();
  summary.textContent = "";
  try {
    const response = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: text, top_k: Number(document.querySelector("#top-k").value) }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(response.status === 422
      ? "Please check your query and result count."
      : "Search is temporarily unavailable. Please try again.");
    if (!Array.isArray(data.results)) throw new Error("The server returned an invalid response.");
    renderResults(data.results);
    const elapsed = ((performance.now() - start) / 1000).toLocaleString("en-US", { maximumFractionDigits: 1 });
    summary.textContent = `${data.results.length} results · ${elapsed}s`;
    status.textContent = data.results.length
      ? `Results for “${text}”. Select a frame to watch the video at that moment.`
      : "No matches found. Try a different or more specific description.";
  } catch (error) {
    status.classList.add("error");
    status.textContent = error instanceof TypeError
      ? "Unable to connect to the server. Please try again." : error.message;
  } finally {
    setLoading(false);
  }
});

query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

document.querySelectorAll(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => { query.value = button.textContent; query.focus(); });
});
