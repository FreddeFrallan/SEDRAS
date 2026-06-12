const leaderboardList = document.querySelector("#leaderboardList");
const modelCount = document.querySelector("#modelCount");
const benchmarkButton = document.querySelector("#benchmarkButton");
const modalBackdrop = document.querySelector("#modalBackdrop");
const closeModalButton = document.querySelector("#closeModalButton");

function formatPercent(value) {
  return `${Math.round(value * 100)}%`;
}

function renderEntry(entry) {
  const statusClass = entry.status.toLowerCase() === "verified" ? " verified" : "";
  return `
    <article class="leaderboard-item">
      <div class="rank">${entry.rank}</div>
      <div>
        <div class="model-name">${entry.model_name}</div>
        <div class="model-meta">${entry.organization} · ${entry.representation}</div>
      </div>
      <div class="stat">
        <span class="stat-label">Accuracy</span>
        <span class="stat-value">${formatPercent(entry.accuracy)}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Full Accuracy</span>
        <span class="stat-value">${formatPercent(entry.full_accuracy)}</span>
      </div>
      <div class="stat">
        <span class="stat-label">Latency</span>
        <span class="stat-value">${entry.latency_seconds.toFixed(1)}s</span>
      </div>
      <div class="stat">
        <span class="stat-label">Status</span>
        <span class="status${statusClass}">${entry.status}</span>
      </div>
    </article>
  `;
}

async function loadLeaderboard() {
  try {
    const response = await fetch("/api/leaderboard");
    if (!response.ok) {
      throw new Error(`Request failed with ${response.status}`);
    }
    const payload = await response.json();
    const entries = payload.entries || [];
    modelCount.textContent = entries.length;
    leaderboardList.innerHTML = entries.map(renderEntry).join("");
  } catch (error) {
    leaderboardList.innerHTML = `<div class="error-row">Could not load leaderboard.</div>`;
  }
}

function openModal() {
  modalBackdrop.hidden = false;
}

function closeModal() {
  modalBackdrop.hidden = true;
}

benchmarkButton.addEventListener("click", openModal);
closeModalButton.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", (event) => {
  if (event.target === modalBackdrop) {
    closeModal();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeModal();
  }
});

loadLeaderboard();
