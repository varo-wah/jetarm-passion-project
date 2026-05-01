# Running The Robot

Run the FastAPI dashboard:

```bash
python3 -m uvicorn jetarm.ui.server_app:app --reload
```

Run the sorting scanner directly:

```bash
python3 -m jetarm.vision.scanner
```

