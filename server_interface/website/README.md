# SEDRAS Benchmark Website

Dependency-free prototype for the public benchmark leaderboard.

## Run

```bash
python -m server_interface.website.backend.app
```

Then open:

```text
http://127.0.0.1:8765
```

## Structure

- `frontend/`: Static HTML, CSS, and JavaScript.
- `backend/`: Python stdlib HTTP server and API routes.
- `database/`: SQLite initialization and seed data.

## API

- `GET /api/leaderboard`: Returns leaderboard rows from SQLite.

